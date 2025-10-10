#!/usr/bin/env python3

import os
from isaacgym import gymapi, gymutil
import torch

def test_model_loading():
    # 初始化gym
    gym = gymapi.acquire_gym()
    
    # 创建仿真参数
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    
    # 创建仿真环境
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        print("Failed to create sim")
        return False
    
    # 创建地面
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)
    
    # 获取资产文件路径
    asset_root = os.path.join(os.path.dirname(__file__), "engineai_gym", "resources")
    asset_file = "robots/Rotunbot/urdf/ball.urdf"
    
    print(f"Asset root: {asset_root}")
    print(f"Asset file: {asset_file}")
    
    # 检查文件是否存在
    full_path = os.path.join(asset_root, asset_file)
    if not os.path.exists(full_path):
        print(f"Asset file not found: {full_path}")
        return False
    
    # 加载资产
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.armature = 0.01
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    
    if asset is None:
        print("Failed to load asset")
        return False
    
    print("Asset loaded successfully!")
    print(f"Asset DOF count: {gym.get_asset_dof_count(asset)}")
    print(f"Asset rigid body count: {gym.get_asset_rigid_body_count(asset)}")
    
    # 清理
    gym.destroy_sim(sim)
    
    return True

if __name__ == "__main__":
    print("Testing Rotunbot model import...")
    if test_model_loading():
        print("Model import test passed!")
    else:
        print("Model import test failed!")