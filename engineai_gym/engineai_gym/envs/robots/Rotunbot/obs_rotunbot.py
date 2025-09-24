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

        return obs_dict

    def get_obs_dict(self):
        """Get the observation dictionary structure.

        Returns:
            dict: Dictionary describing observation structure
        """
        obs_dict = {
            "pendulum_angle": {
                "shape": (2,),  # Two joints: joint1 and joint2
                "type": "pendulum_state"
            },
            "pendulum_angular_vel": {
                "shape": (2,),  # Angular velocities of two joints
                "type": "pendulum_state"
            },
            "base_lin_vel": {
                "shape": (3,),  # x, y, z linear velocities
                "type": "base_state"
            },
            "base_ang_vel": {
                "shape": (3,),  # x, y, z angular velocities
                "type": "base_state"
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
            "pendulum_state": 4,  # pendulum_angle (2) + pendulum_angular_vel (2)
            "base_state": 9,      # base_lin_vel (3) + base_ang_vel (3) + projected_gravity (3)
            "action_history": 2,  # actions (2)
        }

        return obs_sizes

    def get_obs_types(self):
        """Get observation types.

        Returns:
            list: List of observation types
        """
        return ["pendulum_state", "base_state", "action_history"]