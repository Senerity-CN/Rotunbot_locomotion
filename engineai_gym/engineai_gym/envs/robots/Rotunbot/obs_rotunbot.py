import torch
from engineai_gym.envs.base.obs.obs import Obs


class SphericalRobotObs(Obs):
    """Observation class for the spherical robot."""

    def __init__(self, env):
        """Initialize observation handler.

        Args:
            env: Environment instance
        """
        super().__init__(env)

    def compute(self):
        """Compute observations for the current step.

        Returns:
            dict: Dictionary containing observation tensors
        """
        # Get observations from environment
        obs_dict = self.env.compute_observations()
        
        # Apply noise if needed
        if self.env.add_noise:
            # Apply noise to each observation component
            for key in obs_dict:
                if key == "commands":
                    # Commands noise
                    obs_dict[key] += (2 * torch.rand_like(obs_dict[key]) - 1) * self.env.noise_scale_vec[:2]
                elif key == "base_quat":
                    # Quaternion noise
                    obs_dict[key] += (2 * torch.rand_like(obs_dict[key]) - 1) * self.env.noise_scale_vec[2:6]
                elif key == "base_lin_vel":
                    # Linear velocity noise
                    obs_dict[key] += (2 * torch.rand_like(obs_dict[key]) - 1) * self.env.noise_scale_vec[6:9]
                elif key == "base_ang_vel":
                    # Angular velocity noise
                    obs_dict[key] += (2 * torch.rand_like(obs_dict[key]) - 1) * self.env.noise_scale_vec[9:12]
                elif key == "last_base_lin_vel":
                    # Last linear velocity noise
                    obs_dict[key] += (2 * torch.rand_like(obs_dict[key]) - 1) * self.env.noise_scale_vec[12:15]
                elif key == "last_base_ang_vel":
                    # Last angular velocity noise
                    obs_dict[key] += (2 * torch.rand_like(obs_dict[key]) - 1) * self.env.noise_scale_vec[15:18]
                elif key == "dof_pos":
                    # DOF position noise
                    obs_dict[key] += (2 * torch.rand_like(obs_dict[key]) - 1) * self.env.noise_scale_vec[18:19]
                elif key == "dof_vel":
                    # DOF velocity noise
                    obs_dict[key] += (2 * torch.rand_like(obs_dict[key]) - 1) * self.env.noise_scale_vec[19:21]
                # actions and projected_gravity don't have noise according to the noise vector
        
        return obs_dict

    def get_obs_dict(self):
        """Get the observation dictionary structure.

        Returns:
            dict: Dictionary describing observation structure
        """
        obs_dict = {
            "commands": {
                "shape": (2,),  # Linear and angular velocity commands
                "type": "command"
            },
            "base_quat": {
                "shape": (4,),  # Base orientation quaternion
                "type": "base_state"
            },
            "base_lin_vel": {
                "shape": (3,),  # x, y, z linear velocities
                "type": "base_state"
            },
            "base_ang_vel": {
                "shape": (3,),  # x, y, z angular velocities
                "type": "base_state"
            },
            "last_base_lin_vel": {
                "shape": (3,),  # Last x, y, z linear velocities
                "type": "base_state_history"
            },
            "last_base_ang_vel": {
                "shape": (3,),  # Last x, y, z angular velocities
                "type": "base_state_history"
            },
            "dof_pos": {
                "shape": (2,),  # DOF positions
                "type": "joint_state"
            },
            "dof_vel": {
                "shape": (2,),  # DOF velocities
                "type": "joint_state"
            },
            "projected_gravity": {
                "shape": (3,),  # Gravity vector in base frame
                "type": "base_state"
            },
            "actions": {
                "shape": (2,),  # Previous actions for two joints
                "type": "action_history"
            }
        }

        return obs_dict

    def get_obs_sizes(self):
        """Get observation sizes for each observation type.

        Returns:
            dict: Dictionary mapping observation names to their sizes
        """
        obs_sizes = {
            "command": 2,         # commands (2)
            "base_state": 10,     # base_quat (4) + base_lin_vel (3) + base_ang_vel (3)
            "base_state_history": 6,  # last_base_lin_vel (3) + last_base_ang_vel (3)
            "joint_state": 4,     # dof_pos (2) + dof_vel (2)
            "action_history": 2,  # actions (2)
            "gravity": 3,         # projected_gravity (3)
        }

        return obs_sizes

    def get_obs_types(self):
        """Get observation types.

        Returns:
            list: List of observation types
        """
        return ["command", "base_state", "base_state_history", "joint_state", "action_history", "gravity"]

    def commands(self):
        """Get scaled commands observation."""
        # self.obs_scales.command is list, e.g., [1.0, 2.0]
        command_scales_tensor = torch.tensor(self.env.cfg.normalization.obs_scales.command, 
                                           device=self.env.device, dtype=self.env.commands.dtype)
        scaled_commands = self.env.commands[:, :2] * command_scales_tensor
        return scaled_commands

    def base_quat(self):
        """Get base quaternion observation."""
        return self.env.base_quat

    def base_lin_vel(self):
        """Get scaled base linear velocity observation."""
        # self.obs_scales.lin_vel is list, e.g., [0.67, 3.33, 20.0]
        lin_vel_scales_tensor = torch.tensor(self.env.cfg.normalization.obs_scales.lin_vel, 
                                           device=self.env.device, dtype=self.env.base_lin_vel.dtype)
        scaled_base_lin_vel = self.env.base_lin_vel * lin_vel_scales_tensor
        return scaled_base_lin_vel

    def base_ang_vel(self):
        """Get scaled base angular velocity observation."""
        # self.obs_scales.ang_vel is list, e.g., [1.25, 1.25, 1.43]
        ang_vel_scales_tensor = torch.tensor(self.env.cfg.normalization.obs_scales.ang_vel, 
                                           device=self.env.device, dtype=self.env.base_ang_vel.dtype)
        scaled_base_ang_vel = self.env.base_ang_vel * ang_vel_scales_tensor
        return scaled_base_ang_vel

    def last_base_lin_vel(self):
        """Get scaled last base linear velocity observation."""
        # Reuse lin_vel_scales_tensor from base_lin_vel
        lin_vel_scales_tensor = torch.tensor(self.env.cfg.normalization.obs_scales.lin_vel, 
                                           device=self.env.device, dtype=self.env.base_lin_vel.dtype)
        scaled_last_base_lin_vel = self.env.last_base_lin_vel * lin_vel_scales_tensor
        return scaled_last_base_lin_vel

    def last_base_ang_vel(self):
        """Get scaled last base angular velocity observation."""
        # Reuse ang_vel_scales_tensor from base_ang_vel
        ang_vel_scales_tensor = torch.tensor(self.env.cfg.normalization.obs_scales.ang_vel, 
                                           device=self.env.device, dtype=self.env.base_ang_vel.dtype)
        scaled_last_base_ang_vel = self.env.last_base_ang_vel * ang_vel_scales_tensor
        return scaled_last_base_ang_vel

    def dof_pos(self):
        """Get scaled dof position observation."""
        # self.obs_scales.dof_pos is scalar float, e.g., 2.0
        scaled_dof_pos = (self.env.dof_pos[:, 1:2] - self.env.default_dof_pos[:, 1:2]) * self.env.cfg.normalization.obs_scales.dof_pos
        return scaled_dof_pos

    def dof_vel(self):
        """Get scaled dof velocity observation."""
        # self.obs_scales.dof_vel is list, e.g., [0.125, 0.4]
        dof_vel_scales_tensor = torch.tensor(self.env.cfg.normalization.obs_scales.dof_vel, 
                                           device=self.env.device, dtype=self.env.dof_vel.dtype)
        scaled_dof_vel = self.env.dof_vel * dof_vel_scales_tensor
        return scaled_dof_vel

    def projected_gravity(self):
        """Get projected gravity observation."""
        return self.env.projected_gravity

    def actions(self):
        """Get actions observation."""
        return self.env.actions