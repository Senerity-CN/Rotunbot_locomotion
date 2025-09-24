from engineai_rl_workspace.utils.exp_registry import exp_registry
from engineai_rl.runners.on_policy_runner import OnPolicyRunner
from engineai_gym.envs.robots.Rotunbot.config_rotunbot import ConfigRotunbot
from engineai_rl_workspace.exps.Rotunbot.flat.config_rotunbot_flat_ppo import ConfigRotunbotFlatPpo
from engineai_gym.envs.robots.Rotunbot.spherical_robot import SphericalRobot
from engineai_gym.envs.robots.Rotunbot.obs_rotunbot import SphericalRobotObs
from engineai_gym.envs.robots.Rotunbot.rewards_rotunbot import SphericalRobotRewards
from engineai_gym.envs.robots.Rotunbot.domain_rand_rotunbot import SphericalRobotDomainRand
from engineai_rl.algos.ppo import Ppo


# Register the Rotunbot experiment on flat terrain
exp_registry.register(
    name="rotunbot_flat_ppo",
    task_class=SphericalRobot,
    obs_class=SphericalRobotObs,
    domain_rand_class=SphericalRobotDomainRand,
    reward_class=SphericalRobotRewards,
    env_cfg=ConfigRotunbot(),
    runner_class=OnPolicyRunner,
    algo_class=Ppo,
    algo_cfg=ConfigRotunbotFlatPpo(),
)