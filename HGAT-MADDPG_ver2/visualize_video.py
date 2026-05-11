from torch._C import import_ir_module
from yaml import load, Loader
import numpy as np
import json
import pybullet as p
import matplotlib.pyplot as plt
import time
import cv2
import os

from env.scene import Scence

def create_trajectory_video(trajectory_data, output_path="./output/", video_name="trajectory_animation", 
                          fps=30, resolution=(1600, 1600), step_interval=1, add_timestamps=True):
    """
    创建轨迹动画视频
    
    参数:
    - trajectory_data: 轨迹数据
    - output_path: 输出路径
    - video_name: 视频文件名（不含扩展名）
    - fps: 帧率
    - resolution: 视频分辨率 (width, height)
    - step_interval: 步长间隔（每隔几步绘制一帧，可用于控制速度）
    - add_timestamps: 是否添加时间戳
    """
    
    color_list = [
        [1, 0, 0], # 红
        [0.5, 0, 0.5], # 紫
        [1, 0.84, 0] # 土黄
    ]
    
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    video_width, video_height = resolution
    video_filename = os.path.join(output_path, f"{video_name}.mp4")
    
    # 创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (video_width, video_height))

    # 设置相机参数
    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=[0, 0, 22],
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector=[0, 1, 0])
    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=45.0,
        aspect=1.0,
        nearVal=0.1,
        farVal=22.1)

    print(f"开始生成动画，总共约{(len(trajectory_data)-1)//step_interval}帧...")
    
    # 存储已绘制的线段ID，用于可能的清理
    line_ids = []

    # 逐步绘制轨迹并捕获帧
    for i in range(0, len(trajectory_data) - 1, step_interval):
        # 绘制当前段的轨迹
        current_line_ids = []
        for j in range(min(3, len(trajectory_data[i]))):  # 防止索引越界
            line_id = p.addUserDebugLine(
                trajectory_data[i][j], trajectory_data[i+1][j], 
                lineColorRGB=color_list[j % len(color_list)], 
                lifeTime=30000000, lineWidth=5)
            current_line_ids.append(line_id)
        
        line_ids.extend(current_line_ids)
        
        # 让仿真步进
        p.stepSimulation()
        
        # 捕获当前帧
        w, h, rgbPixels, depthPixels, segPixels = p.getCameraImage(
            video_width, video_height,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix)
        
        # 转换图像格式
        rgbPixels = np.array(rgbPixels).reshape(h, w, 4)
        frame = cv2.cvtColor(rgbPixels[:, :, :3], cv2.COLOR_RGB2BGR)
        
        # 添加时间戳或进度信息（可选）
        if add_timestamps:
            progress = (i + 1) / (len(trajectory_data) - 1) * 100
            text = f"Progress: {progress:.1f}% | Frame: {i//step_interval + 1}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 写入视频帧
        video_writer.write(frame.astype(np.uint8))
        
        # 显示进度
        if (i // step_interval + 1) % 50 == 0:
            print(f"已处理 {i//step_interval + 1}/{(len(trajectory_data)-1)//step_interval} 帧")

    # 添加几秒静止帧，让观众看清最终结果
    final_frame = frame.copy()
    for _ in range(fps * 3):  # 3秒静止
        video_writer.write(final_frame)

    # 释放VideoWriter
    video_writer.release()
    print(f"视频保存完成: {video_filename}")
    
    return video_filename, line_ids

if __name__ == "__main__":
    param_path = "/home/star/Yanjun/YJ-MADDPG/HGAT-MADDPG_ver2/config/task.yaml"
    param_dict = load(open(param_path, "r", encoding="utf-8"), Loader=Loader)
    
    # 设置随机数种子
    np.random.seed(param_dict["RANDOM_SEED"])
    cid = p.connect(p.DIRECT)
    scence = Scence()
    scence.construct()
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(
        cameraDistance=9,
        cameraYaw=0,
        cameraPitch=-89,
        cameraTargetPosition=[0, 0, 0]
    )

    # 读取轨迹数据
    file_path = "/home/star/Yanjun/YJ-MADDPG/HGAT-MADDPG_ver2/data/tra/trajectory_2854.json"
    with open(file_path, "r") as file:
        trajectory_data = file.read()
    trajectory_data = np.array((json.loads(trajectory_data)))

    # 创建视频 - 可以调整这些参数
    video_file, line_ids = create_trajectory_video(
        trajectory_data=trajectory_data,
        output_path="./output/",
        video_name="trajectory_animation_enhanced",
        fps=24,  # 降低帧率以减少文件大小
        resolution=(1280, 1280),  # 稍小的分辨率
        step_interval=2,  # 每隔2步绘制一帧，加快动画速度
        add_timestamps=True
    )
    
    # 可选：创建不同视角的视频
    # 重置视角并创建另一个视频
    p.resetDebugVisualizerCamera(
        cameraDistance=15,
        cameraYaw=45,
        cameraPitch=-60,
        cameraTargetPosition=[0, 0, 0]
    )
    
    # 清除之前的线段
    for line_id in line_ids:
        p.removeUserDebugItem(line_id)
    
    # 创建侧视角视频
    video_file_side, _ = create_trajectory_video(
        trajectory_data=trajectory_data,
        output_path="./output/",
        video_name="trajectory_animation_side_view",
        fps=24,
        resolution=(1280, 1280),
        step_interval=2,
        add_timestamps=True
    )
    
    print("所有视频生成完成！")
    time.sleep(5)