import os
import torch
import numpy as np
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from engineai_gym.envs.base.legged_robot import LeggedRobot
from engineai_rl_lib.math import quat_apply_yaw, wrap_to_pi, get_euler_xyz_tensor
from .config_rotunbot import ConfigRotunbot


class SphericalRobot(LeggedRobot):
    """Spherical robot environment for locomotion training."""

    def __init__(
        self,
        obs_class,
        goal_class,
        domain_rand_class,
        reward_class,
        cfg: ConfigRotunbot,
        sim_params,
        physics_engine,
        sim_device,
        headless,
    ):
        """Initialize the spherical robot environment.

        Args:
            cfg (ConfigRotunbot): Environment configuration
            sim_params (gymapi.SimParams): Simulation parameters
            physics_engine (gymapi.SimType): Physics engine type
            sim_device (str): Simulation device
            headless (bool): Whether to run in headless mode
        """
        super().__init__(
            obs_class,
            goal_class,
            domain_rand_class,
            reward_class,
            cfg,
            sim_params,
            physics_engine,
            sim_device,
            headless,
        )
        
        # 添加球形机器人特有的状态变量
        self.last_base_lin_vel = torch.zeros_like(self.base_lin_vel)
        self.last_base_ang_vel = torch.zeros_like(self.base_ang_vel)
        self.step_counter = 0
        self.t = 0

    def _parse_cfg(self):
        """Parse configuration parameters."""
        super()._parse_cfg()
        
        self.command_x_range = self.cfg.commands.ranges.lin_vel_x
        self.command_y_range = self.cfg.commands.ranges.lin_vel_y
        self.command_yaw_range = self.cfg.commands.ranges.ang_vel_yaw
        
        self.asset_file = self.cfg.asset.file.format(
            ENGINEAI_GYM_PACKAGE_DIR=os.getenv("ENGINEAI_GYM_PACKAGE_DIR", "")
        )
        self.obs_scales = {}

    def _create_envs(self):
        """Create robot environments."""
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        # Load robot asset
        asset_root = os.path.dirname(self.asset_file)
        asset_file = os.path.basename(self.asset_file)
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # Get asset information
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        
        # Set num_dof for compatibility with LeggedRobot
        self.num_dof = self.num_dofs

        # Prepare environment creation
        env_lower = gymapi.Vec3(-self.cfg.env.env_spacing, -self.cfg.env.env_spacing, 0.0)
        env_upper = gymapi.Vec3(self.cfg.env.env_spacing, self.cfg.env.env_spacing, self.cfg.env.env_spacing)

        # Create environments
        self.envs = []
        self.actor_handles = []

        for i in range(self.num_envs):
            # Create environment
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))

            # Set initial pose
            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(*self.cfg.init_state.pos)
            # Set rotation from config
            start_pose.r = gymapi.Quat(*self.cfg.init_state.rot)

            # Create actor
            actor_handle = self.gym.create_actor(env_ptr, robot_asset, start_pose, "rotunbot", i, 0, 0)
            self.actor_handles.append(actor_handle)

            # Set initial dof positions
            default_dof_pos = np.array([self.cfg.init_state.default_joint_angles[name]
                                      for name in self.cfg.env.action_joints])
            default_dof_pos = to_torch(default_dof_pos, device=self.device)

            # Create DofState array for set_actor_dof_states
            num_dof = len(self.cfg.env.action_joints)
            dof_state = np.zeros(num_dof, dtype=gymapi.DofState.dtype)
            dof_state[:]['pos'] = default_dof_pos.cpu().numpy()

            # Set actor dof states
            self.gym.set_actor_dof_states(env_ptr, actor_handle, dof_state, gymapi.STATE_ALL)

            # Add environment to list
            self.envs.append(env_ptr)
        
        # Initialize foot indices (needed by wrappers)
        # For spherical robot, we use link1 as the foot
        self.foot_indices = torch.zeros(1, dtype=torch.long, device=self.device)
        self.foot_indices[0] = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], "link1"
        )
        
        # Set action joint indices for compatibility with LeggedRobot
        self.action_joint_indices = list(range(len(self.cfg.env.action_joints)))
        
        # Set body names for compatibility with LeggedRobot
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        
        # Initialize min_joint_armature for compatibility with LeggedRobot
        self.min_joint_armature = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)

    def _resample_commands(self, env_ids):
        """Resample movement commands for spherical robot."""
        if self.t < 200:
            self.commands[env_ids, 0] = self.t / 200
        else:
            self.commands[env_ids, 0] = 1
        
        self.t = self.t + 1

        # 目标角速度
        target_ang_vel_yaw = 0

        # 当前角速度
        current_ang_vel_yaw = self.base_ang_vel[env_ids, 2]

        # 限制每次更新的最大变化幅度
        max_delta = 0.1  # 最大变化幅度
        delta = target_ang_vel_yaw - current_ang_vel_yaw
        delta = torch.clamp(delta, min=-max_delta, max=max_delta)  # 限制变化幅度

        # 更新角速度指令
        self.commands[env_ids, 1] = current_ang_vel_yaw + delta

        # 限制角速度指令
        yaw_limit = torch.abs(self.commands[env_ids, 0]/2)
        for i in range(len(env_ids)):
            if self.commands[env_ids[i], 1] > yaw_limit[i]:
                self.commands[env_ids[i], 1] = yaw_limit[i]
            if self.commands[env_ids[i], 1] < -yaw_limit[i]:
                self.commands[env_ids[i], 1] = -yaw_limit[i]

    def post_physics_step(self):
        """Check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations 
        calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        
        # Compute observations after reset
        self.obs_dict = self.compute_observations()
        self.goal_dict = self.compute_goals()

        # Update last states
        self.last_actions[:] = self.actions[:]
        self.last_base_lin_vel = self.base_lin_vel.clone()
        self.last_base_ang_vel = self.base_ang_vel.clone()
        self.last_dof_vel = self.dof_vel.clone()
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def compute_observations(self):
        """Computes observations for the spherical robot"""
        obs_dict = {}
        
        # 1. commands (线速度和角速度)
        command_scales_tensor = torch.tensor(self.obs_scales.get("commands", [1.0, 2.0]), device=self.device, dtype=self.commands.dtype)
        scaled_commands = self.commands[:, :2] * command_scales_tensor
        obs_dict["commands"] = scaled_commands
        
        # 2. base_quat (基座四元数)
        obs_dict["base_quat"] = self.base_quat
        
        # 3. base_lin_vel (线速度)
        lin_vel_scales_tensor = torch.tensor(self.obs_scales.get("base_lin_vel", [0.67, 3.33, 20.0]), device=self.device, dtype=self.base_lin_vel.dtype)
        scaled_base_lin_vel = self.base_lin_vel * lin_vel_scales_tensor
        obs_dict["base_lin_vel"] = scaled_base_lin_vel
        
        # 4. base_ang_vel (角速度)
        ang_vel_scales_tensor = torch.tensor(self.obs_scales.get("base_ang_vel", [1.25, 1.25, 1.43]), device=self.device, dtype=self.base_ang_vel.dtype)
        scaled_base_ang_vel = self.base_ang_vel * ang_vel_scales_tensor
        obs_dict["base_ang_vel"] = scaled_base_ang_vel
        
        # 5. last_base_lin_vel (上一时刻线速度)
        scaled_last_base_lin_vel = self.last_base_lin_vel * lin_vel_scales_tensor
        obs_dict["last_base_lin_vel"] = scaled_last_base_lin_vel
        
        # 6. last_base_ang_vel (上一时刻角速度)
        scaled_last_base_ang_vel = self.last_base_ang_vel * ang_vel_scales_tensor
        obs_dict["last_base_ang_vel"] = scaled_last_base_ang_vel
        
        # 7. dof_pos (电机角度 - 副轴)
        # For spherical robot, we only use the second joint (steering)
        scaled_dof_pos = (self.dof_pos[:, 1:2] - self.default_dof_pos[:, 1:2]) * self.obs_scales.get("dof_pos", 2.0)
        obs_dict["dof_pos"] = scaled_dof_pos
        
        # 8. dof_vel (电机速度)
        dof_vel_scales_tensor = torch.tensor(self.obs_scales.get("dof_vel", [0.125, 0.4]), device=self.device, dtype=self.dof_vel.dtype)
        scaled_dof_vel = self.dof_vel * dof_vel_scales_tensor
        obs_dict["dof_vel"] = scaled_dof_vel
        
        # 9. projected_gravity (重力)
        obs_dict["projected_gravity"] = self.projected_gravity
        
        # 10. actions (动作历史)
        obs_dict["actions"] = self.actions
        
        return {"non_lagged_obs": obs_dict, "lagged_obs": {}}

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.step_counter += 1
        
        # Clip actions
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        
        # Scale actions for P and V control
        if self.cfg.control.control_type == "P and V":
            actions_scaled = torch.zeros_like(actions)
            actions_scaled[:, 0] = torch.clip(actions[:, 0], -8, 8) * self.cfg.control.first_actionScale
            actions_scaled[:, 1] = torch.clip(actions[:, 1], -0.5236, 0.5236) * self.cfg.control.second_actionScale
            self.actions = actions_scaled
        else:
            # Scale actions
            self.actions = actions * self.cfg.control.action_scale
            
        # Store last states
        self.last_actions = self.actions.clone()
        self.last_dof_vel = self.dof_vel.clone()
        self.last_dof_pos = self.dof_pos.clone()
        self.last_base_lin_vel = self.base_lin_vel.clone()
        self.last_base_ang_vel = self.base_ang_vel.clone()

        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques().view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        
        self.post_physics_step()
        
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        # Since we don't have obs_buf in this implementation, we'll return the observation dict
        # In a real implementation, you would process the observations here
        return self.obs_dict, self.goal_dict, self.rew_buf, self.reset_buf, self.extras

    def _compute_torques(self):
        """Compute torques from actions for spherical robot using P and V control.

        Returns:
            torch.Tensor: Torques sent to the simulation
        """
        control_type = self.cfg.control.control_type
        try:
            actions = self.domain_rands.domain_rands_type_action_lag.lagged_actions
        except:
            actions = self.actions
            
        actions_scaled = actions * self.action_scales
        self.controller_input[:, self.action_joint_indices] = actions_scaled
        
        try:
            motor_offsets = self.domain_rands.domain_rands_type_dof.motor_offsets
        except:
            motor_offsets = 0
        try:
            coulomb_friction = self.domain_rands.domain_rands_type_dof.coulomb_friction
        except:
            coulomb_friction = 0
            
        if control_type == "P":
            torques = (
                self.p_gains
                * (
                    self.controller_input
                    + self.default_dof_pos
                    - self.dof_pos
                    + motor_offsets
                )
                - self.d_gains * self.dof_vel
                - coulomb_friction
            )
        elif control_type == "V":
            torques = (
                self.p_gains * (self.controller_input - self.dof_vel)
                - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
                - coulomb_friction
            )
        elif control_type == "T":
            torques = self.controller_input.clone()
        elif control_type == "P and V":
            # For spherical robot, we have two joints with different control strategies
            torques = torch.zeros_like(self.dof_pos)
            # First joint (driving): velocity control
            torques[:, 0] = self.p_gains[0] * (actions_scaled[:, 0] - self.dof_vel[:, 0]) - self.d_gains[0] * (self.dof_vel[:, 0] - self.last_dof_vel[:, 0]) / self.sim_params.dt
            # Second joint (steering): position control
            torques[:, 1] = self.p_gains[1] * (actions_scaled[:, 1] + self.default_dof_pos[:, 1] - self.dof_pos[:, 1]) - self.d_gains[1] * self.dof_vel[:, 1]
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        try:
            torque_multi = self.domain_rands.domain_rands_type_dof.torque_multi
        except:
            torque_multi = 1
        torques *= torque_multi
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    # ------------ reward functions ----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xyz axes)
        error_x = self.commands[:, 0] - self.base_lin_vel[:, 0]
        error_y = self.base_lin_vel[:, 1]  # 实际侧向速度，目标为0
        error_z = self.base_lin_vel[:, 2]  # 实际垂直速度，目标为0 (可选)

        # 计算总的平方误差
        lin_vel_error = torch.square(error_x) + torch.square(error_y) + torch.square(error_z)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_lin_vel_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        target_yaw_vel = self.commands[:, 1]
        actual_yaw_vel = self.base_ang_vel[:, 2]

        ang_vel_error = torch.square(target_yaw_vel - actual_yaw_vel)
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_ang_vel_sigma)