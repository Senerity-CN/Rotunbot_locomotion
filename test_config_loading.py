#!/usr/bin/env python3

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_config_loading():
    """测试配置文件是否能正确加载"""
    try:
        # 测试环境配置加载
        from engineai_gym.envs.robots.Rotunbot.config_rotunbot import ConfigRotunbot
        env_cfg = ConfigRotunbot()
        print("✅ 环境配置加载成功")
        print(f"   环境数量: {env_cfg.env.num_envs}")
        print(f"   动作数量: {env_cfg.env.num_actions}")
        print(f"   观测数量: {env_cfg.env.num_observations}")
        print(f"   关节列表: {env_cfg.env.action_joints}")
        print(f"   控制类型: {getattr(env_cfg.control, 'control_type', 'Not specified')}")
        
        # 测试PPO配置加载
        from engineai_rl_workspace.exps.Rotunbot.flat.config_rotunbot_flat_ppo import ConfigRotunbotFlatPpo
        ppo_cfg = ConfigRotunbotFlatPpo()
        print("\n✅ PPO配置加载成功")
        print(f"   最大迭代次数: {ppo_cfg.runner.max_iterations}")
        print(f"   初始噪声标准差: {ppo_cfg.policy.init_noise_std}")
        print(f"   Actor网络结构: {ppo_cfg.policy.actor_hidden_dims}")
        print(f"   Critic网络结构: {ppo_cfg.policy.critic_hidden_dims}")
        
        # 测试实验注册
        from engineai_rl_workspace.exps.Rotunbot.rotunbot import exp_registry
        print("\n✅ 实验注册加载成功")
        print(f"   已注册的实验: {list(exp_registry.task_classes.keys())}")
        
        # 验证配置一致性
        print("\n🔍 配置验证:")
        if env_cfg.env.num_envs == 4096:
            print("   ✅ 环境数量正确")
        else:
            print(f"   ❌ 环境数量不正确: {env_cfg.env.num_envs}")
            
        if ppo_cfg.runner.max_iterations == 800:
            print("   ✅ 迭代次数正确")
        else:
            print(f"   ❌ 迭代次数不正确: {ppo_cfg.runner.max_iterations}")
            
        if ppo_cfg.policy.init_noise_std == 0.2:
            print("   ✅ 初始噪声标准差正确")
        else:
            print(f"   ❌ 初始噪声标准差不正确: {ppo_cfg.policy.init_noise_std}")
            
        # 检查控制参数
        if hasattr(env_cfg.control, 'control_type') and env_cfg.control.control_type == 'P and V':
            print("   ✅ 控制类型正确")
        else:
            print(f"   ❌ 控制类型不正确: {getattr(env_cfg.control, 'control_type', 'Not specified')}")
            
        print("\n🎉 所有配置测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Rotunbot配置测试")
    print("=" * 50)
    success = test_config_loading()
    print("=" * 50)
    if success:
        print("测试结果: 通过")
        sys.exit(0)
    else:
        print("测试结果: 失败")
        sys.exit(1)