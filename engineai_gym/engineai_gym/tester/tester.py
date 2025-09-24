import numpy as np
import os
import yaml
from engineai_gym import ENGINEAI_GYM_ROOT_DIR
from engineai_gym.tester.testers.tester_type_base import TesterTypeBase
from engineai_rl_lib.class_operations import (
    instance_name_to_class_name,
    add_space_to_class_name,
)
from engineai_gym.wrapper.record_video_wrapper import RecordVideoWrapper
from engineai_rl_lib.files_and_dirs import import_modules_of_specific_type_from_path
from engineai_rl_lib.class_operations import get_class_and_parent_paths
from engineai_rl_lib.dict_operations import expand_and_overwrite_dict
import inspect


class Tester:
    def __init__(self, length, save_path, tester_config_path):
        self.length = length
        self.save_path = save_path
        with open(tester_config_path) as file:
            self.tester_config = yaml.safe_load(file)

    def set_env_cfg(self, env_cfg):
        env_cfg.domain_rands.rigid_shape.randomize_friction = False
        env_cfg.domain_rands.rigid_shape.randomize_restitution = False
        env_cfg.domain_rands.rigid_body.randomize_base_mass = False
        env_cfg.domain_rands.rigid_body.randomize_com = False
        env_cfg.domain_rands.rigid_body.randomize_link_mass = False
        env_cfg.domain_rands.rigid_body.randomize_inertia = False
        env_cfg.domain_rands.dof.randomize_gains = False
        env_cfg.domain_rands.dof.randomize_torque = False
        env_cfg.domain_rands.dof.randomize_motor_offset = False
        env_cfg.domain_rands.dof.randomize_joint_friction = False
        env_cfg.domain_rands.dof.randomize_joint_armature = False
        env_cfg.domain_rands.dof.randomize_coulomb_friction = False
        env_cfg.domain_rands.action_lag.action_lag_timesteps = 0
        env_cfg.domain_rands.action_lag.randomize_action_lag_timesteps = False
        env_cfg.domain_rands.obs_lag.motor_lag_timesteps = 0
        env_cfg.domain_rands.obs_lag.randomize_motor_lag_timesteps = False
        env_cfg.domain_rands.obs_lag.imu_lag_timesteps = 0
        env_cfg.domain_rands.obs_lag.randomize_imu_lag_timesteps = False
        env_cfg.domain_rands.disturbance.push_robots = False
        return env_cfg

    def set_env(self, env, record_video=False):
        self.env = env
        self.record_video = record_video
        if record_video:
            self.env_supports_video_recording = isinstance(env, RecordVideoWrapper)
            if self.env_supports_video_recording:
                if self.env.record_length > self.length:
                    raise ValueError(
                        "Video record length must be less than the tester length!"
                    )
            else:
                raise ValueError("Env doesn't support video recording!")

    def init_testers(self, dt, save_path, extra_args=None):
        exec(f"from {self.__class__.__module__} import {self.__class__.__name__}")
        files = get_class_and_parent_paths(self.__class__, Tester)
        files.reverse()
        files.insert(0, inspect.getfile(Tester))
        self.imported_classes = {}
        for file in files:
            current_file_directory = os.path.join(os.path.dirname(file), "testers")
            if os.path.exists(current_file_directory):
                self.imported_classes = expand_and_overwrite_dict(
                    self.imported_classes,
                    import_modules_of_specific_type_from_path(
                        ENGINEAI_GYM_ROOT_DIR, current_file_directory, TesterTypeBase
                    ),
                )
        time = np.linspace(0, self.length * dt, self.length)
        if extra_args is None:
            extra_args = {}

        self.testers = {}
        for key, value in self.tester_config["testers"].items():
            loggers = {}
            for logger in value["loggers"]:
                loggers[logger.replace("logger_type_", "")] = logger
            tester_class = self.imported_classes[key]
            name = add_space_to_class_name(instance_name_to_class_name(key))
            self.testers[key] = tester_class(
                name,
                files,
                loggers,
                self.env,
                time,
                os.path.join(save_path, "data"),
                extra_args,
            )
        self.tester_names = list(self.testers.keys())
        return self.testers

    def retrieve_data_from_loggers(self, idx):
        tester = self.get_current_tester(idx)
        if (idx + 1) % self.length == 0:
            for logger in tester.loggers:
                logger.retrieve_data()
                logger.log.clear()

    def add_data_for_testers_to_log(self, idx, extra_data):
        tester = self.get_current_tester(idx)
        for logger in tester.loggers:
            logger.log_data(extra_data)

    def set_goals(self, idx):
        return self.get_current_tester(idx).set_goals()

    def get_current_tester(self, idx):
        if idx // self.length < self.num_testers:
            return self.testers[self.tester_names[idx // self.length]]
        else:
            raise RuntimeError("tester is not found!")

    def process_record_video(self, idx):
        tester = self.get_current_tester(idx)
        if idx % self.length == 0:
            tester.start_record_video()
        elif (idx + 1) % self.env.record_length == 0 and self.env.is_recording_video:
            tester.end_and_save_recording_video()

    def step(self, idx, extra_data):
        if idx % self.length == 0:
            print(f"\nStart tester: {self.tester_names[idx // self.length]}")
        self.add_data_for_testers_to_log(idx, extra_data)
        self.retrieve_data_from_loggers(idx)
        if self.record_video:
            self.process_record_video(idx)

    @property
    def num_testers(self):
        return len(self.testers)
