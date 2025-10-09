# Rotunbot模型迁移说明

本文档总结了将Rotunbot模型从legged_gym仓库迁移到当前仓库的工作。

## 迁移状态

✅ **已完成**: 模型文件迁移
✅ **已完成**: 目录结构创建
✅ **已完成**: URDF和网格文件复制
✅ **已完成**: 测试脚本创建
✅ **已完成**: 更改提交和推送
✅ **已完成**: 配置文件迁移
✅ **已完成**: 奖励函数迁移
✅ **已完成**: 控制算法迁移
✅ **已完成**: 观测计算迁移
✅ **已完成**: 域随机化迁移

## 添加的文件

1. **模型文件**:
   - `engineai_gym/resources/robots/Rotunbot/urdf/ball.urdf` - 主URDF模型文件
   - `engineai_gym/resources/robots/Rotunbot/meshes/base_link.STL` - 基座链接网格
   - `engineai_gym/resources/robots/Rotunbot/meshes/link1.STL` - 主球形体网格
   - `engineai_gym/resources/robots/Rotunbot/meshes/link2.STL` - 副轴网格

2. **配置文件**:
   - `engineai_gym/engineai_gym/envs/robots/Rotunbot/config_rotunbot.py` - 环境配置文件
   - `engineai_rl_workspace/exps/Rotunbot/flat/config_rotunbot_flat_ppo.py` - PPO配置文件

3. **核心实现文件**:
   - `engineai_gym/engineai_gym/envs/robots/Rotunbot/spherical_robot.py` - 环境主类
   - `engineai_gym/engineai_gym/envs/robots/Rotunbot/rewards_rotunbot.py` - 奖励函数类
   - `engineai_gym/engineai_gym/envs/robots/Rotunbot/obs_rotunbot.py` - 观测计算类
   - `engineai_gym/engineai_gym/envs/robots/Rotunbot/domain_rand_rotunbot.py` - 域随机化类

4. **测试脚本**:
   - `test_model_import.py` - 用于验证Isaac Gym中模型加载的脚本
   - `test_config_loading.py` - 用于验证配置文件加载的脚本

## 迁移详情

### 1. 模型文件
- 将Rotunbot的URDF文件和网格文件从源仓库复制到目标仓库
- 更新了文件路径引用以匹配目标仓库结构

### 2. 环境配置
- 迁移了详细的环境配置，包括:
  - 26维观测空间配置
  - P和V组合控制参数
  - 命令范围和重采样时间
  - 归一化参数
  - 噪声配置
  - 奖励缩放参数

### 3. PPO配置
- 迁移了PPO算法配置，包括:
  - 网络结构: [256, 128, 64]
  - 初始噪声: 0.2
  - 最大迭代次数: 800

### 4. 奖励函数
- 实现了源仓库中的所有奖励函数:
  - 线速度跟踪奖励
  - 角速度跟踪奖励
  - Z轴线速度惩罚
  - XY轴角速度惩罚
  - 力矩惩罚
  - 关节加速度惩罚
  - 动作变化率惩罚

### 5. 控制算法
- 实现了P和V组合控制算法
- 支持多种控制模式 (P, V, T, P and V)
- 根据配置文件中的PD参数计算力矩

### 6. 观测计算
- 实现了26维观测空间计算
- 包括命令、四元数、速度、历史信息等
- 支持观测噪声添加

### 7. 域随机化
- 实现了摩擦系数随机化
- 实现了恢复系数随机化
- 实现了质量随机化 (基础质量和链接质量)
- 实现了阻尼随机化

## 下一步工作

1. **完整功能验证**:
   - 在安装了Isaac Gym的机器上运行完整测试
   - 验证所有组件协同工作

2. **训练测试**:
   - 运行训练脚本验证端到端功能
   - 检查日志和奖励曲线

3. **性能优化**:
   - 根据需要调整超参数
   - 优化训练性能

## 目录结构

```
engineai_gym/resources/robots/Rotunbot/
├── urdf/
│   └── ball.urdf
└── meshes/
    ├── base_link.STL
    ├── link1.STL
    └── link2.STL
```

## 机器人描述

Rotunbot是一个球形机器人，具有:
- 3个链接: base_link, link1(球形体), link2(副轴)
- 2个关节: 
  - joint1: 连续关节，用于主球形运动
  - joint2: 旋转关节，用于副轴转向
- 用于视觉表示的网格文件
- URDF中定义的物理属性

## 配置详情

### 环境配置
- 环境数量: 4096
- 观测空间: 26维，包含命令、四元数、速度、历史信息等
- 动作空间: 2维，控制两个关节
- 控制方式: P和V组合控制
- 命令范围: 
  - 线速度: [-1.0, 1.0] m/s
  - 角速度: [-0.5, 0.5] rad/s

### PPO配置
- 网络结构: [256, 128, 64]
- 初始噪声: 0.2
- 最大迭代次数: 800
- 学习率: 1.e-3

## 测试

### 模型导入测试
```bash
python test_model_import.py
```

此脚本将:
1. 初始化Isaac Gym仿真
2. 加载Rotunbot模型
3. 报告资产信息
4. 清理资源

### 配置加载测试
```bash
python test_config_loading.py
```

此脚本将:
1. 加载环境配置
2. 加载PPO配置
3. 验证关键参数
4. 报告测试结果

测试应在正确安装了Isaac Gym的机器上运行。