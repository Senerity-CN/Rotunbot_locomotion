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
        # Compute individual reward components
        tracking_lin_vel_reward = self._reward_tracking_lin_vel()
        tracking_ang_vel_reward = self._reward_tracking_ang_vel()
        lin_vel_z_reward = self._reward_lin_vel_z()
        ang_vel_xy_reward = self._reward_ang_vel_xy()
        torques_reward = self._reward_torques()
        dof_acc_reward = self._reward_dof_acc()
        action_rate_reward = self._reward_action_rate()
        
        # Scale rewards according to configuration
        total_reward = (
            self.env.cfg.rewards.scales.tracking_lin_vel * tracking_lin_vel_reward +
            self.env.cfg.rewards.scales.tracking_ang_vel * tracking_ang_vel_reward +
            self.env.cfg.rewards.scales.lin_vel_z * lin_vel_z_reward +
            self.env.cfg.rewards.scales.ang_vel_xy * ang_vel_xy_reward +
            self.env.cfg.rewards.scales.torques * torques_reward +
            self.env.cfg.rewards.scales.dof_acc * dof_acc_reward +
            self.env.cfg.rewards.scales.action_rate * action_rate_reward
        )
        
        # Apply positive rewards clipping if configured
        if self.env.cfg.rewards.only_positive_rewards:
            total_reward = torch.clamp(total_reward, min=0.0)
        
        # Store individual reward components for logging
        self.reward_components = {
            "tracking_lin_vel": tracking_lin_vel_reward,
            "tracking_ang_vel": tracking_ang_vel_reward,
            "lin_vel_z": lin_vel_z_reward,
            "ang_vel_xy": ang_vel_xy_reward,
            "torques": torques_reward,
            "dof_acc": dof_acc_reward,
            "action_rate": action_rate_reward,
            "total": total_reward
        }
        
        return total_reward

    def _reward_tracking_lin_vel(self):
        """Tracking of linear velocity commands (xy axes)"""
        # Tracking of linear velocity commands (xyz axes)
        error_x = self.env.commands[:, 0] - self.env.base_lin_vel[:, 0]
        error_y = self.env.base_lin_vel[:, 1]  # 实际侧向速度，目标为0
        error_z = self.env.base_lin_vel[:, 2]  # 实际垂直速度，目标为0 (可选)
        
        # 计算总的平方误差
        lin_vel_error = torch.square(error_x) + torch.square(error_y) + torch.square(error_z)  # 加上z轴惩罚的版本
        return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_lin_vel_sigma)

    def _reward_tracking_ang_vel(self):
        """Tracking of angular velocity commands (yaw)"""
        # Tracking of angular velocity commands (yaw)
        target_yaw_vel = self.env.commands[:, 1]
        actual_yaw_vel = self.env.base_ang_vel[:, 2]
        
        ang_vel_error = torch.square(target_yaw_vel - actual_yaw_vel)
        tracking_sigma_for_ang = self.env.cfg.rewards.tracking_ang_vel_sigma
        
        return torch.exp(-ang_vel_error / tracking_sigma_for_ang)

    def _reward_lin_vel_z(self):
        """Penalize z axis base linear velocity"""
        return torch.square(self.env.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        """Penalize xy axes base angular velocity"""
        return torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)

    def _reward_torques(self):
        """Penalize torques"""
        return torch.sum(torch.square(self.env.torques), dim=1)

    def _reward_dof_acc(self):
        """Penalize dof accelerations"""
        # 计算关节加速度 (当前关节速度 - 上一时刻关节速度) / dt
        dof_acc = (self.env.dof_vel - self.env.last_dof_vel) / self.env.sim_params.dt
        return torch.sum(torch.square(dof_acc), dim=1)

    def _reward_action_rate(self):
        """Penalize changes in actions"""
        # 计算动作变化率 (当前动作 - 上一时刻动作)
        action_rate = torch.sum(torch.square(self.env.actions - self.env.last_actions), dim=1)
        return action_rate

    def get_reward_names(self):
        """Get names of reward components.

        Returns:
            list: List of reward component names
        """
        return [
            "tracking_lin_vel",
            "tracking_ang_vel",
            "lin_vel_z",
            "ang_vel_xy",
            "torques",
            "dof_acc",
            "action_rate",
            "total"
        ]