from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject, ArticulationCfg
from isaaclab.utils.math import quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def body_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """The feet contact of the robot."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.001

def object_position_in_robot_frame(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """计算指令中的目标位置在机器人底盘坐标系下的表达。"""
    # 1. 获取机器人资产
    asset: ArticulationCfg = env.scene[asset_cfg.name]
    
    # 2. 获取世界坐标系下的目标位置 (来自你的 ee_pose command)
    # 假设 command 返回的是目标点的世界坐标 [num_envs, 3]
    target_pos_w = env.command_manager.get_command(command_name)[:, :3]
    
    # 3. 获取底盘的世界位姿
    base_pos_w = asset.data.root_pos_w
    base_quat_w = asset.data.root_quat_w
    
    # 4. 将世界坐标转换到底盘局部坐标系
    # 使用 isaaclab.utils.math 中的 subtract_frame 或类似逻辑
    relative_pos_w = target_pos_w - base_pos_w
    target_pos_b = quat_rotate_inverse(base_quat_w, relative_pos_w)
    
    return target_pos_b