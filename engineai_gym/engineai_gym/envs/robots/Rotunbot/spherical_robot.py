import os
import torch
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from engineai_gym.envs.base.env_base import EnvBase
from engineai_rl_lib.math import quat_apply_yaw, wrap_to_pi, get_euler_xyz_tensor
from .config_rotunbot import ConfigRotunbot


class SphericalRobot(EnvBase):
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
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False

        # Parse configuration
        self._parse_cfg()

        # Initialize base environment
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

    def _parse_cfg(self):
        """Parse configuration parameters."""
        # Initialize command ranges
        self.command_x_range = self.cfg.commands.ranges.lin_vel_x
        self.command_y_range = self.cfg.commands.ranges.lin_vel_y
        self.command_yaw_range = self.cfg.commands.ranges.ang_vel_yaw

        # Store asset path
        self.asset_file = self.cfg.asset.file.format(
            ENGINEAI_GYM_PACKAGE_DIR=os.getenv("ENGINEAI_GYM_PACKAGE_DIR", "")
        )

    def create_sim(self):
        """Create simulation environment."""
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params
        )
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        # Create terrain
        self._create_ground_plane()

        # Create environments
        self._create_envs()

    def _create_ground_plane(self):
        """Create ground plane for simulation."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        """Create robot environments."""
        # Define asset options
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_base_link = self.cfg.asset.replace_base_link
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
        robot_asset = self.gym.load_asset(self.sim, self.asset_file, asset_options)

        # Get asset information
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)

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
            start_pose.r = gymapi.Quat(*self.cfg.init_state.rot)

            # Create actor
            actor_handle = self.gym.create_actor(env_ptr, robot_asset, start_pose, "rotunbot", i, 0, 0)
            self.actor_handles.append(actor_handle)

            # Set initial dof positions
            default_dof_pos = np.array([self.cfg.init_state.default_joint_angles[name]
                                      for name in self.cfg.env.action_joints])
            default_dof_pos = to_torch(default_dof_pos, device=self.device)

            # Set actor dof states
            self.gym.set_actor_dof_states(env_ptr, actor_handle, default_dof_pos, gymapi.STATE_ALL)

            # Add environment to list
            self.envs.append(env_ptr)

        # Initialize buffers
        self._init_buffers()

    def _init_buffers(self):
        """Initialize tensor buffers."""
        # Get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        # Refresh tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Create tensor wrappers
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_force_tensor).view(
            self.num_envs, -1, 3
        )

        # Split dof state
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # Initialize other buffers
        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.commands_scale = torch.tensor(
            [self.command_x_range[1] - self.command_x_range[0],
             self.command_y_range[1] - self.command_y_range[0],
             self.command_yaw_range[1] - self.command_yaw_range[0]],
            device=self.device
        )

        # Initialize pendulum-specific buffers
        self.pendulum_angle = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.pendulum_angular_vel = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)

    def compute_observations(self):
        """Compute observations for the policy."""
        # Update pendulum state from dof positions and velocities
        self.pendulum_angle = self.dof_pos.clone()
        self.pendulum_angular_vel = self.dof_vel.clone()

        # Get base velocity in base frame
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        # Project gravity to base frame
        gravity_vec = to_torch([0., 0., -1.], device=self.device).repeat((self.num_envs, 1))
        self.projected_gravity = quat_rotate_inverse(self.base_quat, gravity_vec)

        # Create observation dictionary
        obs_dict = {
            "pendulum_angle": self.pendulum_angle,
            "pendulum_angular_vel": self.pendulum_angular_vel,
            "base_lin_vel": self.base_lin_vel,
            "base_ang_vel": self.base_ang_vel,
            "projected_gravity": self.projected_gravity,
            "actions": self.actions,
        }

        return obs_dict

    def compute_reward(self):
        """Compute reward for current step."""
        # This will be implemented in the reward class
        pass

    def reset_idx(self, env_ids):
        """Reset environments."""
        if len(env_ids) == 0:
            return

        # Reset pendulum positions
        positions = torch_rand_float(
            self.cfg.init_state.default_joint_angles["joint1"] - 0.1,
            self.cfg.init_state.default_joint_angles["joint1"] + 0.1,
            (len(env_ids), 1),
            device=self.device
        ).squeeze(1)

        velocities = torch_rand_float(
            -0.1, 0.1, (len(env_ids), 1), device=self.device
        ).squeeze(1)

        self.dof_pos[env_ids, 0] = positions
        self.dof_vel[env_ids, 0] = velocities

        # Reset second joint (steering)
        positions = torch_rand_float(
            self.cfg.init_state.default_joint_angles["joint2"] - 0.1,
            self.cfg.init_state.default_joint_angles["joint2"] + 0.1,
            (len(env_ids), 1),
            device=self.device
        ).squeeze(1)

        velocities = torch_rand_float(
            -0.1, 0.1, (len(env_ids), 1), device=self.device
        ).squeeze(1)

        self.dof_pos[env_ids, 1] = positions
        self.dof_vel[env_ids, 1] = velocities

        # Apply resets
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

        # Reset root states
        self.root_states[env_ids] = self.base_init_state
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

        # Resample commands
        self._resample_commands(env_ids)

    def _resample_commands(self, env_ids):
        """Resample movement commands."""
        # Sample linear velocity commands
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device
        ).squeeze()

        self.commands[env_ids, 1] = torch_rand_float(
            self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device
        ).squeeze()

        # Sample angular velocity commands
        self.commands[env_ids, 2] = torch_rand_float(
            self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device
        ).squeeze()

    def pre_physics_step(self, actions):
        """Process actions before physics step."""
        # Scale actions
        self.actions = actions * self.cfg.control.action_scale

        # Apply forces to joints
        forces = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)

        # Apply force to joint1 (forward/backward motion)
        forces[:, 0] = self.actions[:, 0] * 10.0  # Scale factor for force

        # Apply force to joint2 (steering)
        forces[:, 1] = self.actions[:, 1] * 5.0   # Scale factor for steering force

        # Apply forces
        self.gym.apply_actor_dof_efforts(self.envs, self.actor_handles, forces)

    def post_physics_step(self):
        """Process after physics step."""
        # Refresh tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Update episode buffers
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # Compute observations, rewards, and check termination
        self.compute_observations()
        self.compute_reward()
        self.check_termination()

        # Reset environments if needed
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)