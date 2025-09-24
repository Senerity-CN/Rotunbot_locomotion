import os, asyncio, multiprocessing
import pygame
from threading import Thread
import numpy as np
from tqdm import tqdm
from git import Repo

from engineai_gym import ENGINEAI_GYM_PACKAGE_DIR, ENGINEAI_GYM_ROOT_DIR
from engineai_rl_workspace.utils import (
    get_args,
    generate_cfg_files_from_json,
)
from engineai_rl_workspace.utils.process_resume_files import (
    get_log_root_and_log_dir,
    checkout_resume_commit,
)
from engineai_rl_workspace import (
    REDIS_HOST,
    LOCK_KEY,
    REDIS_PORT,
    LOCK_TIMEOUT,
    LOCK_MESSAGE,
    INITIALIZATION_COMPLETE_MESSAGE,
    ENGINEAI_WORKSPACE_ROOT_DIR,
    FAIL_TO_LOAD_JSON_MESSAGE,
)
from engineai_rl_lib.redis_lock import RedisLock
from engineai_rl_lib.git import (
    get_current_commit_and_branch,
    checkout_commit_or_branch,
    unstash_files,
    apply_patch,
    stash_files,
    unstash_files_without_removing,
)
from engineai_rl_lib.files_and_dirs import import_attr_from_file_path


async def play(args):
    global lock, repo, current_commit, current_branch
    lock = RedisLock(REDIS_HOST, LOCK_KEY, REDIS_PORT, LOCK_TIMEOUT, LOCK_MESSAGE)
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd, still_cmd, last_x_vel_cmd, last_y_vel_cmd, last_yaw_vel_cmd, last_still_cmd
    (
        x_vel_cmd,
        y_vel_cmd,
        yaw_vel_cmd,
        still_cmd,
        last_x_vel_cmd,
        last_y_vel_cmd,
        last_yaw_vel_cmd,
        last_still_cmd,
    ) = (0, 0, 0, False, 0, 0, 0, False)
    args.resume = True
    if not await lock.acquire():
        print("Could not acquire lock, exiting...")
        return
    repo = Repo(ENGINEAI_WORKSPACE_ROOT_DIR)
    current_commit, current_branch = get_current_commit_and_branch(repo)
    if not args.current_files:
        _, log_dir = get_log_root_and_log_dir(args)
        checkout_resume_commit(log_dir, repo)
        apply_patch(os.path.join(log_dir, "resume.patch"), ENGINEAI_WORKSPACE_ROOT_DIR)
    else:
        stash_files(repo)
        unstash_files_without_removing(repo)

    process = multiprocessing.Process(target=generate_cfg_files_from_json, args=(args,))
    process.start()
    process.join()
    if process.exitcode != 0:
        if lock.redis.get(lock.lock_key) == lock.pid.encode():
            try:
                checkout_commit_or_branch(repo, current_commit, current_branch)
                unstash_files(repo)
            finally:
                lock.release()
        raise RuntimeError(FAIL_TO_LOAD_JSON_MESSAGE)
    from engineai_gym.wrapper import VecGymWrapper, RecordVideoWrapper
    import engineai_rl_workspace.exps
    from engineai_rl_workspace.utils.exp_registry import exp_registry

    (
        args,
        task_class,
        obs_class,
        goal_class,
        domain_rand_class,
        reward_class,
        runner_class,
        algo_class,
        log_dir,
        log_root,
        env_cfg,
        algo_cfg,
    ) = exp_registry.get_class_and_cfg(name=args.exp_name, args=args)
    if not args.late_restore:
        checkout_commit_or_branch(repo, current_commit, current_branch)
        unstash_files(repo)
        if lock.redis.get(lock.lock_key) == lock.pid.encode():
            lock.release()
        print(INITIALIZATION_COMPLETE_MESSAGE)

    # override some parameters for testing
    if args.use_joystick:
        env_cfg.env.num_envs = 1
    tester_class = import_attr_from_file_path(
        ENGINEAI_GYM_ROOT_DIR,
        env_cfg.tester.class_path.format(
            ENGINEAI_GYM_PACKAGE_DIR=ENGINEAI_GYM_PACKAGE_DIR
        ),
        env_cfg.tester.class_name,
    )
    tester = tester_class(
        args.test_length,
        os.path.join(log_dir, "test"),
        env_cfg.tester.config_path.format(
            ENGINEAI_GYM_PACKAGE_DIR=ENGINEAI_GYM_PACKAGE_DIR
        ),
    )
    env_cfg = tester.set_env_cfg(env_cfg)

    # prepare environment
    env = exp_registry.make_env(
        task_class,
        obs_class,
        goal_class,
        domain_rand_class,
        reward_class,
        args,
        env_cfg,
    )
    env = VecGymWrapper(env)
    if args.video:
        env = RecordVideoWrapper(
            env,
            manual=True,
            frame_size=args.frame_size,
            fps=args.fps,
            record_interval=args.record_interval,
            record_length=args.record_length,
            num_steps_per_env=algo_cfg.runner.num_steps_per_env,
            env_idx=args.env_idx_record,
            actor_idx=args.actor_idx_record,
            rigid_body_idx=args.rigid_body_idx_record,
            camera_offset=args.camera_offset,
            camera_rotation=args.camera_rotation,
            video_path=os.path.join(log_dir, "test", "videos"),
        )

    if args.late_restore:
        checkout_commit_or_branch(repo, current_commit, current_branch)
        unstash_files(repo)
        if lock.redis.get(lock.lock_key) == lock.pid.encode():
            lock.release()
        print(INITIALIZATION_COMPLETE_MESSAGE)
    tester.set_env(env, args.video)
    tester.init_testers(
        env.dt,
        os.path.join(log_dir, "test"),
        extra_args={"robot_index": args.env_idx_record},
    )
    # load policy
    runner = exp_registry.make_alg_runner(env, args.exp_name, args, log_dir)
    policy = runner.get_inference_policy()

    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    if args.use_joystick:
        iteration_range = range(100000)
    else:
        iteration_range = tqdm(range(tester.num_testers * args.test_length))
    if args.use_joystick:
        inputs = runner.reset(
            set_commands_from_joystick,
            set_goals_callback_args=(env, x_vel_cmd, y_vel_cmd, yaw_vel_cmd, still_cmd),
        )
    else:
        inputs = runner.reset(tester.set_goals, set_goals_callback_args=(0,))
    for iter in iteration_range:
        if args.use_joystick:
            inputs, actions, _, _, _ = runner.step(
                inputs,
                policy,
                set_commands_from_joystick,
                set_goals_callback_args=(
                    env,
                    x_vel_cmd,
                    y_vel_cmd,
                    yaw_vel_cmd,
                    still_cmd,
                ),
            )
        else:
            if iter + 1 < tester.num_testers * args.test_length:
                inputs, actions, _, _, _ = runner.step(
                    inputs,
                    policy,
                    tester.set_goals,
                    set_goals_callback_args=(iter + 1,),
                )
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)
        if not args.use_joystick:
            tester.step(iter, {"actions": actions})


def set_commands_from_joystick(env, x_vel_cmd, y_vel_cmd, yaw_vel_cmd, still_cmd):
    global last_x_vel_cmd, last_y_vel_cmd, last_yaw_vel_cmd, last_still_cmd
    env.vel_commands[:, 0] = x_vel_cmd
    env.vel_commands[:, 1] = y_vel_cmd
    env.vel_commands[:, 2] = yaw_vel_cmd
    env.still_commands[:] = still_cmd
    if (
        last_x_vel_cmd != x_vel_cmd
        or last_y_vel_cmd != y_vel_cmd
        or last_yaw_vel_cmd != yaw_vel_cmd
        or last_still_cmd != still_cmd
    ):
        print(
            "Current Command: \n",
            "Vel Commands: ",
            env.vel_commands,
            "Still Commands: ",
            env.still_commands,
        )
        last_x_vel_cmd = x_vel_cmd
        last_y_vel_cmd = y_vel_cmd
        last_yaw_vel_cmd = yaw_vel_cmd
        last_still_cmd = still_cmd


def use_joystick(args):
    joystick_opened = False
    pygame.init()
    try:
        # get joystick
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_opened = True
    except Exception as e:
        print(f"Unable to turn on joystick：{e}")

    # handle joystick thread
    def handle_joystick_input():
        global x_vel_cmd, y_vel_cmd, yaw_vel_cmd, still_cmd, last_x_vel_cmd, last_y_vel_cmd, last_yaw_vel_cmd, last_still_cmd
        while True:
            # get joystick input
            pygame.event.get()

            # update command
            x_vel_cmd = -joystick.get_axis(1) * args.joystick_scale[0]
            y_vel_cmd = -joystick.get_axis(0) * args.joystick_scale[1]
            yaw_vel_cmd = -joystick.get_axis(3) * args.joystick_scale[2]
            if joystick.get_button(0):
                still_cmd = ~still_cmd

            # wait for a short period of time
            pygame.time.delay(100)

    # start thread
    if joystick_opened:
        joystick_thread = Thread(target=handle_joystick_input)
        joystick_thread.start()


if __name__ == "__main__":
    global lock, repo, current_commit, current_branch
    try:
        MOVE_CAMERA = False
        args = get_args()
        if args.use_joystick:
            use_joystick(args)
        asyncio.run(play(args))
    except KeyboardInterrupt or SystemExit:
        if lock.redis.get(lock.lock_key) == lock.pid.encode():
            try:
                checkout_commit_or_branch(repo, current_commit, current_branch)
                unstash_files(repo)
            finally:
                lock.release()
