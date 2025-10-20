#!/usr/bin/env python3

"""
Test script for Rotunbot motion in Isaac Gym.
This script loads the Rotunbot and applies a fixed velocity to link1 while keeping link2 at 0.
"""

import os
import numpy as np
from isaacgym import gymapi, gymtorch
from engineai_gym.envs.robots.Rotunbot.config_rotunbot import ConfigRotunbot
import torch

def main():
    # 初始化Gym
    gym = gymapi.acquire_gym()
    
    # 创建仿真参数
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    
    # 物理引擎设置
    physics_engine = gymapi.SIM_PHYSX
    sim = gym.create_sim(0, -1, physics_engine, sim_params)
    
    if sim is None:
        print("*** Failed to create sim")
        quit()
    
    # 创建地面平面
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)
    
    # 加载Rotunbot资产
    asset_root = "/home/balance/Rotunbot_locomotion/engineai_gym/resources/robots/Rotunbot/urdf"
    asset_file = "ball.urdf"
    
    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_VEL
    asset_options.fix_base_link = False
    asset_options.collapse_fixed_joints = False
    
    rotunbot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    
    if rotunbot_asset is None:
        print("*** Failed to load asset")
        quit()
    
    # 获取资产信息
    num_dof = gym.get_asset_dof_count(rotunbot_asset)
    print(f"Number of DOF: {num_dof}")
    
    # 创建环境
    env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
    
    # 设置初始姿态
    start_pose = gymapi.Transform()
    start_pose.p = gymapi.Vec3(0.0, 0.0, 0.5)  # 初始位置
    start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # 初始旋转
    
    # 创建actor
    actor_handle = gym.create_actor(env, rotunbot_asset, start_pose, "rotunbot", 0, 1)
    
    # 设置DOF属性
    dof_props = gym.get_actor_dof_properties(env, actor_handle)
    dof_props['driveMode'][:].fill(gymapi.DOF_MODE_VEL)
    dof_props['stiffness'][:].fill(0.0)
    dof_props['damping'][:].fill(10.0)
    gym.set_actor_dof_properties(env, actor_handle, dof_props)
    
    # 获取状态张量
    dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
    
    # 刷新张量
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    
    # 包装为PyTorch张量
    dof_states = gymtorch.wrap_tensor(dof_state_tensor)
    root_states = gymtorch.wrap_tensor(root_state_tensor)
    
    # 设置固定速度 (rad/s)
    fixed_velocity_link1 = 5.0  # link1的固定速度
    fixed_velocity_link2 = 0.0   # link2始终保持为0
    
    # 记录初始位置
    initial_root_pos = None
    initial_root_rot = None
    
    # 创建速度命令
    dof_velocities = torch.zeros(num_dof, dtype=torch.float32, device='cpu')
    dof_velocities[0] = fixed_velocity_link1  # 设置link1速度
    dof_velocities[1] = fixed_velocity_link2  # 设置link2速度
    
    # 仿真循环
    print("Starting simulation. Press Ctrl+C to exit.")
    print("Link1 velocity:", fixed_velocity_link1, "rad/s")
    print("Link2 velocity:", fixed_velocity_link2, "rad/s")
    print("-" * 50)
    
    try:
        frame_count = 0
        while True:
            # 每100帧打印一次状态
            if frame_count % 10000 == 0:
                gym.refresh_dof_state_tensor(sim)
                gym.refresh_actor_root_state_tensor(sim)
                
                # 获取当前DOF状态
                current_dof_pos = dof_states[:, 0]  # 位置
                current_dof_vel = dof_states[:, 1]  # 速度
                
                # 获取根状态
                root_pos = root_states[0, :3]  # 位置 x, y, z
                root_rot = root_states[0, 3:7] # 四元数 x, y, z, w
                root_lin_vel = root_states[0, 7:10] # 线速度
                root_ang_vel = root_states[0, 10:13] # 角速度
                
                # 记录初始位置
                if initial_root_pos is None:
                    initial_root_pos = root_pos.clone()
                    initial_root_rot = root_rot.clone()
                
                # 计算位置变化
                pos_change = root_pos - initial_root_pos if initial_root_pos is not None else torch.zeros_like(root_pos)
                
                print(f"Frame {frame_count}:")
                print(f"  DOF Positions: [{current_dof_pos[0]:.3f}, {current_dof_pos[1]:.3f}]")
                print(f"  DOF Velocities: [{current_dof_vel[0]:.3f}, {current_dof_vel[1]:.3f}]")
                print(f"  Root Position: [{root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f}]")
                print(f"  Position Change: [{pos_change[0]:.3f}, {pos_change[1]:.3f}, {pos_change[2]:.3f}]")
                print(f"  Root Linear Velocity: [{root_lin_vel[0]:.3f}, {root_lin_vel[1]:.3f}, {root_lin_vel[2]:.3f}]")
                print(f"  Root Angular Velocity: [{root_ang_vel[0]:.3f}, {root_ang_vel[1]:.3f}, {root_ang_vel[2]:.3f}]")
                print()
            
            # 应用速度控制
            gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(dof_velocities))
            
            # 仿真一步
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            
            # 渲染
            gym.step_graphics(sim)
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    
    # 清理
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main()