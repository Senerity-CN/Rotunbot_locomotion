# Rotunbot 球形机器人训练指南

## 简介

Rotunbot是一个球形机器人，通过内部摆锤的摆动实现运动。它有两个关节：
- `joint1` (continuous类型)：控制球形机器人前进后退
- `joint2` (revolute类型)：控制球形机器人转向

## 训练命令

### 基本训练
```bash
# 使用默认配置训练Rotunbot在平坦地形上的运动
python engineai_rl_workspace/scripts/train.py --exp_name rotunbot_flat_ppo
```

### 恢复训练
```bash
# 从之前的检查点恢复训练
python engineai_rl_workspace/scripts/train.py --exp_name rotunbot_flat_ppo --resume
```

### 自定义训练参数
```bash
# 设置特定的训练参数
python engineai_rl_workspace/scripts/train.py --exp_name rotunbot_flat_ppo --num_envs 2048 --max_iterations 3000
```

## 配置文件说明

### 环境配置 (`config_rotunbot.py`)
- `num_envs`: 并行环境数量
- `obs_list`: 观测空间包含摆锤角度、角速度、基座速度等
- `action_joints`: 控制关节(joint1和joint2)
- `terrain`: 地形设置(平面、粗糙等)

### 算法配置 (`config_rotunbot_flat_ppo.py`)
- `num_steps_per_env`: 每个环境的步数
- `max_iterations`: 最大训练迭代次数
- `learning_rate`: 学习率
- `entropy_coef`: 熵系数

## 观测空间

观测空间包含以下信息：
1. `pendulum_angle`: 摆锤角度(2个关节)
2. `pendulum_angular_vel`: 摆锤角速度(2个关节)
3. `base_lin_vel`: 基座线速度(x, y, z)
4. `base_ang_vel`: 基座角速度(x, y, z)
5. `projected_gravity`: 重力投影向量
6. `actions`: 上一时刻的动作

## 奖励函数

奖励函数包含以下组件：
1. 速度跟踪奖励：跟踪目标速度
2. 能量效率奖励：惩罚过大的动作
3. 稳定性奖励：惩罚过大的摆锤角度
4. 平滑性奖励：惩罚过大的摆锤角速度变化
5. 前进奖励：鼓励向前运动
6. 倒下惩罚：机器人倒下时的惩罚

## 领域随机化

包括以下参数的随机化：
1. 摩擦系数
2. 恢复系数
3. 质量
4. 阻尼