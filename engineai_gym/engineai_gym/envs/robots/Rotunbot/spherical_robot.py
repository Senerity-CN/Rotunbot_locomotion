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
        # Store cfg for later use
        self.cfg = cfg
        self._parse_cfg()
        
        # Initialize base environment using LeggedRobot
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
        
        # Action scales will be set by LeggedRobot

    def _parse_cfg(self):
        """Parse configuration parameters."""
        # Initialize command ranges
        self.command_x_range = self.cfg.commands.ranges.lin_vel_x
        self.command_y_range = self.cfg.commands.ranges.lin_vel_y
        self.command_yaw_range = self.cfg.commands.ranges.ang_vel_yaw
        
        # Episode length will be set by LeggedRobot after sim_params is available
        
        # Store asset path
        self.asset_file = self.cfg.asset.file.format(
            ENGINEAI_GYM_PACKAGE_DIR=os.getenv("ENGINEAI_GYM_PACKAGE_DIR", "")
        )

    def _create_envs(self):
        """Create robot environments."""
        # Define asset options
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
        """Resample movement commands."""
        # Sample linear velocity commands
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device
        ).squeeze()

        # Sample angular velocity commands
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device
        ).squeeze()

    def pre_physics_step(self, actions):
        """Process actions before physics step."""
        # Store last states
        self.last_actions = self.actions.clone()
        self.last_dof_vel = self.dof_vel.clone()
        self.last_dof_pos = self.dof_pos.clone()
        self.last_base_lin_vel = self.base_lin_vel.clone()
        self.last_base_ang_vel = self.base_ang_vel.clone()

        # Scale actions for P and V control
        if self.cfg.control.control_type == "P and V":
            self.actions = torch.zeros_like(actions)
            self.actions[:, 0] = actions[:, 0] * self.cfg.control.first_actionScale
            self.actions[:, 1] = actions[:, 1] * self.cfg.control.second_actionScale
        else:
            # Scale actions
            self.actions = actions * self.cfg.control.action_scale

    def _compute_torques(self):
        """Compute torques from actions using P and V control.

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
            torques = deepcopy(self.controller_input)
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