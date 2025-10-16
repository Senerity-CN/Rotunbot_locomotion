import os
import torch
import numpy as np
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

        # Initialize buffers after simulation is created
        self._init_buffers()
        
        # Create action_scales property for compatibility with wrappers
        self.action_scales = {
            "joint1": self.cfg.control.first_actionScale,
            "joint2": self.cfg.control.second_actionScale
        }

    def _parse_cfg(self):
        """Parse configuration parameters."""
        # Initialize command ranges
        self.command_x_range = self.cfg.commands.ranges.lin_vel_x
        self.command_y_range = self.cfg.commands.ranges.lin_vel_y
        self.command_yaw_range = self.cfg.commands.ranges.ang_vel_yaw
        
        # Initialize time step
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        
        # Initialize episode length
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

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
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)

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
            # Normalize quaternion to ensure valid rotation
            rot_quat = self.cfg.init_state.rot
            rot_norm = np.linalg.norm(rot_quat)
            if rot_norm > 0:
                rot_quat = [q / rot_norm for q in rot_quat]
            start_pose.r = gymapi.Quat(*rot_quat)

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
        # Ensure the contact forces tensor has the correct shape
        contact_force_tensor = gymtorch.wrap_tensor(net_contact_force_tensor)
        if contact_force_tensor.numel() > 0:
            self.contact_forces = contact_force_tensor.view(self.num_envs, -1, 3)
        else:
            self.contact_forces = torch.zeros(self.num_envs, 1, 3, dtype=torch.float, device=self.device)

        # Split dof state
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # Initialize other buffers
        self.commands = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)
        self.last_dof_vel = torch.zeros_like(self.dof_vel, device=self.device)
        self.last_dof_pos = torch.zeros_like(self.dof_pos, device=self.device)

        # Initialize PD gains
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        
        # Set PD gains from config
        for i, joint_name in enumerate(self.cfg.env.action_joints):
            self.p_gains[i] = self.cfg.control.stiffness.get(joint_name, 0.0)
            self.d_gains[i] = self.cfg.control.damping.get(joint_name, 0.0)

        # Initialize base state
        self.base_init_state = torch.zeros(self.num_envs, 13, dtype=torch.float, device=self.device)
        self.base_init_state[:, :3] = to_torch(self.cfg.init_state.pos, device=self.device)
        self.base_init_state[:, 3:7] = to_torch(self.cfg.init_state.rot, device=self.device)
        self.base_init_state[:, 7:10] = to_torch(self.cfg.init_state.lin_vel, device=self.device)
        self.base_init_state[:, 10:13] = to_torch(self.cfg.init_state.ang_vel, device=self.device)

        # Initialize default dof positions
        self.default_dof_pos = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)
        for i, joint_name in enumerate(self.cfg.env.action_joints):
            self.default_dof_pos[:, i] = self.cfg.init_state.default_joint_angles.get(joint_name, 0.0)

        # Initialize other variables
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch([0., 0., -1.], device=self.device).repeat((self.num_envs, 1))
        
        # Initialize base velocity and other required attributes
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        # Initialize last state variables for observations
        self.last_base_lin_vel = torch.zeros_like(self.base_lin_vel, device=self.device)
        self.last_base_ang_vel = torch.zeros_like(self.base_ang_vel, device=self.device)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13], device=self.device)
        
        # Initialize DOF limits (needed by wrappers)
        dof_props = self.gym.get_actor_dof_properties(self.envs[0], self.actor_handles[0])
        self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device)
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        
        for i in range(self.num_dof):
            self.dof_pos_limits[i, 0] = dof_props["lower"][i].item()
            self.dof_pos_limits[i, 1] = dof_props["upper"][i].item()
            self.dof_vel_limits[i] = dof_props["velocity"][i].item()
            self.torque_limits[i] = dof_props["effort"][i].item()
            
        # Initialize observation scales
        self.obs_scales = {}
        for obs_name in self.cfg.env.obs_list:
            # Handle the case where obs_scales is a class instead of a dict
            if hasattr(self.cfg.normalization.obs_scales, '__dict__'):
                # It's a class, get the attribute directly
                if hasattr(self.cfg.normalization.obs_scales, obs_name):
                    obs_scale_value = getattr(self.cfg.normalization.obs_scales, obs_name)
                    if isinstance(obs_scale_value, list):
                        obs_scales_tensor = torch.zeros(
                            self.num_dof, device=self.device, dtype=torch.float
                        )
                        # For list values, we need to map them to joints
                        for idx, joint_name in enumerate(self.dof_names):
                            # Simple mapping: use the first value for all joints for now
                            # You might want to customize this mapping based on your needs
                            obs_scales_tensor[idx] = obs_scale_value[0] if obs_scale_value else 1.0
                        self.obs_scales[obs_name] = obs_scales_tensor
                    else:
                        self.obs_scales[obs_name] = obs_scale_value
                else:
                    self.obs_scales[obs_name] = 1.0
            else:
                # It's a dict, use the original approach
                if isinstance(self.cfg.normalization.obs_scales.get(obs_name, 1), dict):
                    obs_scales_tensor = torch.zeros(
                        self.num_dof, device=self.device, dtype=torch.float
                    )
                    for idx, joint_name in enumerate(self.dof_names):
                        for (
                            joint_type,
                            obs_scale,
                        ) in self.cfg.normalization.obs_scales.get(obs_name).items():
                            if joint_type in joint_name:
                                obs_scales_tensor[idx] = obs_scale
                    self.obs_scales[obs_name] = obs_scales_tensor
                else:
                    self.obs_scales[obs_name] = self.cfg.normalization.obs_scales.get(
                        obs_name, 1
                    )
            
        # Initialize dictionaries (needed by wrappers)
        self.obs_dict = {}
        self.goal_dict = {}
        
        # Initialize velocity commands (needed by wrappers)
        self.vel_commands = torch.zeros(
            self.num_envs,
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # x vel, y vel, yaw vel
        
        # Initialize still commands (needed by wrappers)
        self.still_commands = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )

    def compute_observations(self):
        """Compute observations for the policy."""
        # Create observation dictionary with proper structure
        obs_data = {
            "commands": self.commands * self.obs_scales.get("commands", 1),
            "base_quat": self.base_quat * self.obs_scales.get("base_quat", 1),
            "base_lin_vel": self.base_lin_vel * self.obs_scales.get("base_lin_vel", 1),
            "base_ang_vel": self.base_ang_vel * self.obs_scales.get("base_ang_vel", 1),
            "last_base_lin_vel": self.last_base_lin_vel * self.obs_scales.get("last_base_lin_vel", 1),
            "last_base_ang_vel": self.last_base_ang_vel * self.obs_scales.get("last_base_ang_vel", 1),
            "dof_pos": self.dof_pos * self.obs_scales.get("dof_pos", 1),
            "dof_vel": self.dof_vel * self.obs_scales.get("dof_vel", 1),
            "projected_gravity": self.projected_gravity * self.obs_scales.get("projected_gravity", 1),
            "actions": self.actions * self.obs_scales.get("actions", 1),
        }
        
        # Structure expected by wrapper: before_reset and after_reset
        non_lagged_obs_dict = {
            "before_reset": obs_data,
            "after_reset": obs_data
        }
        
        # For spherical robot, we don't have lagged observations, so return empty dict
        lagged_obs_dict = {
            "before_reset": {},
            "after_reset": {}
        }

        return {"non_lagged_obs": non_lagged_obs_dict, "lagged_obs": lagged_obs_dict}

    def compute_reward(self):
        """Compute reward for current step."""
        # This will be implemented in the reward class
        reward = self.rewards.compute_reward()
        self.rew_buf = reward
        return reward

    def compute_goals(self):
        """Compute goals for the policy."""
        # For spherical robot, the goal is the commands
        return {
            "commands": self.commands
        }

    def reset_idx(self, env_ids):
        """Reset environments."""
        if len(env_ids) == 0:
            return

        # Reset dof positions and velocities
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
        self.root_states[env_ids] = self.base_init_state[env_ids]
        # Ensure proper rotation quaternion normalization
        self.root_states[env_ids, 3:7] = self.root_states[env_ids, 3:7] / torch.norm(self.root_states[env_ids, 3:7], dim=1, keepdim=True)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

        # Resample commands
        self._resample_commands(env_ids)

        # Reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_dof_pos[env_ids] = 0.
        self.last_base_lin_vel[env_ids] = 0.
        self.last_base_ang_vel[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = False

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

        # Scale actions
        self.actions = actions * self.cfg.control.action_scale

        # Compute torques based on control type
        self.torques = self._compute_torques(self.actions).view(self.torques.shape)

        # Apply forces to joints
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

    def _compute_torques(self, actions):
        """Compute torques from actions using P and V control.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            torch.Tensor: Torques sent to the simulation
        """
        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = self.p_gains * (actions + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        elif control_type == "V":
            torques = self.p_gains * (actions - self.dof_vel) - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
            torques = actions
        elif control_type == "P and V":
            actions_scaled = torch.zeros_like(actions)  # 或者 actions.clone()
            actions_scaled[:, 0] = actions[:, 0] * self.cfg.control.first_actionScale
            actions_scaled[:, 1] = actions[:, 1] * self.cfg.control.second_actionScale

            torques = torch.zeros_like(self.dof_pos)
            torques[:, 0] = self.p_gains[0] * (actions_scaled[:, 0] - self.dof_vel[:, 0]) - self.d_gains[0] * (self.dof_vel[:, 0] - self.last_dof_vel[:, 0]) / self.sim_params.dt
            torques[:, 1] = self.p_gains[1] * (actions_scaled[:, 1] + self.default_dof_pos[:, 1] - self.dof_pos[:, 1]) - self.d_gains[1] * self.dof_vel[:, 1]
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)

    def post_physics_step(self):
        """Process after physics step."""
        # Refresh tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Update base velocities and other state variables
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        # Update episode buffers
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # Compute observations, rewards, and check termination
        self.obs_dict = self.compute_observations()
        self.compute_reward()
        self.check_termination()

        # Compute goals
        self.goal_dict = self.compute_goals()

        # Reset environments if needed
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

    def check_termination(self):
        """Check if environments need to be reset."""
        # Reset if episode length is exceeded
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

        # Reset if robot falls (z position too low)
        self.reset_buf |= (self.root_states[:, 2] < 0.2)

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.

        Args:
            cfg (Dict): Environment config file

        Returns:
            torch.Tensor: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(self.cfg.env.num_observations, dtype=torch.float, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:2] = 0.  # commands
        noise_vec[2:6] = noise_scales.quat * noise_level
        noise_vec[6:9] = noise_scales.lin_vel * noise_level
        noise_vec[9:12] = noise_scales.ang_vel * noise_level
        noise_vec[12:15] = noise_scales.lin_vel * noise_level
        noise_vec[15:18] = noise_scales.ang_vel * noise_level
        noise_vec[18:19] = noise_scales.dof_pos * noise_level
        noise_vec[19:21] = noise_scales.dof_vel * noise_level
        noise_vec[21:26] = 0.  # previous actions
        return noise_vec

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # Clip actions
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        
        # Step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.pre_physics_step(self.actions)
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.post_physics_step()

        return self.obs_dict, self.goal_dict, self.rew_buf, self.reset_buf, self.extras