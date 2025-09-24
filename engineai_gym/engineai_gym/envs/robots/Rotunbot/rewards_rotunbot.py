import torch
from engineai_gym.envs.base.rewards.rewards import Rewards


class SphericalRobotRewards(Rewards):
    """Reward class for the spherical robot."""

    def __init__(self, env):
        """Initialize reward handler.

        Args:
            env: Environment instance
        """
        super().__init__(env)
        self.env = env

    def compute_reward(self):
        """Compute reward for current step.

        Returns:
            torch.Tensor: Reward tensor of shape (num_envs,)
        """
        # Extract relevant states
        base_lin_vel = self.env.base_lin_vel
        base_ang_vel = self.env.base_ang_vel
        actions = self.env.actions
        pendulum_angle = self.env.pendulum_angle
        pendulum_angular_vel = self.env.pendulum_angular_vel

        # 1. Velocity tracking reward
        # Reward for tracking commanded linear velocity
        lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        lin_vel_reward = torch.exp(-lin_vel_error / 0.25)

        # Reward for tracking commanded angular velocity
        ang_vel_error = torch.square(self.env.commands[:, 2] - base_ang_vel[:, 2])
        ang_vel_reward = torch.exp(-ang_vel_error / 0.25)

        # 2. Energy efficiency reward
        # Penalize large actions to encourage energy efficiency
        action_penalty = torch.sum(torch.square(actions), dim=1)
        energy_reward = -0.01 * action_penalty

        # 3. Stability reward
        # Penalize large pendulum angles to encourage stability
        pendulum_stability_penalty = torch.sum(torch.square(pendulum_angle), dim=1)
        stability_reward = -0.1 * pendulum_stability_penalty

        # 4. Smoothness reward
        # Penalize large changes in pendulum angular velocity
        pendulum_smoothness_penalty = torch.sum(torch.square(pendulum_angular_vel), dim=1)
        smoothness_reward = -0.01 * pendulum_smoothness_penalty

        # 5. Forward progression reward
        # Additional reward for forward movement
        forward_reward = 0.5 * base_lin_vel[:, 0]

        # 6. Penalty for falling
        # Detect if robot has fallen (z position too low)
        fall_penalty = (self.env.root_states[:, 2] < 0.2).float() * -10.0

        # Combine all rewards
        total_reward = (
            lin_vel_reward +
            ang_vel_reward +
            energy_reward +
            stability_reward +
            smoothness_reward +
            forward_reward +
            fall_penalty
        )

        # Store individual reward components for logging
        self.reward_components = {
            "lin_vel_reward": lin_vel_reward,
            "ang_vel_reward": ang_vel_reward,
            "energy_reward": energy_reward,
            "stability_reward": stability_reward,
            "smoothness_reward": smoothness_reward,
            "forward_reward": forward_reward,
            "fall_penalty": fall_penalty,
        }

        return total_reward

    def get_reward_names(self):
        """Get names of reward components.

        Returns:
            list: List of reward component names
        """
        return [
            "lin_vel_reward",
            "ang_vel_reward",
            "energy_reward",
            "stability_reward",
            "smoothness_reward",
            "forward_reward",
            "fall_penalty"
        ]