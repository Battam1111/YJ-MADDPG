#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils.py 模块 - 模型加载和动作归一化工具函数
===================================================

本模块主要提供以下功能：
1. 获取最新检查点目录（latest_logdir）：遍历指定目录下的所有运行记录，并返回最新运行的完整路径。
2. 获取加载模型路径（get_load_path）：支持用户自定义模型路径加载，
   如果指定了 custom_model_path 参数，则直接使用该路径（经过存在性检查），
   否则按照默认逻辑在最新运行目录中查找最新模型文件。
3. 动作归一化函数（action_normalize）：将输入动作张量归一化为单位向量。

作者：YourName
日期：202X-XX-XX
"""

import os
import torch

def latest_logdir(log_root):
    """
    获取日志目录中最新的运行目录

    参数:
        log_root (str): 日志根目录路径，例如 "hcanet-3.27_maddpg/data/logs/"
    
    返回:
        str: 最新运行目录的完整路径

    异常:
        如果日志根目录中不存在任何子目录，则会抛出 ValueError 异常。
    """
    # 检查日志根目录是否存在
    if not os.path.exists(log_root):
        raise ValueError("日志根目录不存在: " + log_root)
    
    # 列出日志根目录下的所有子目录
    target_runs = [run for run in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, run))]
    
    if not target_runs:
        raise ValueError("日志目录中没有任何运行目录: " + log_root)
    
    # 按名称排序（假设目录名称中含有日期时间信息，排序后最新的在最后）
    target_runs.sort()
    latest_run = target_runs[-1]
    print(f'found the latest logdir: {latest_run}')
    return os.path.join(log_root, latest_run)

def get_load_path(root, custom_model_path=None):
    """
    获取模型加载的完整路径。

    功能说明：
    1. 如果提供了 custom_model_path（自定义模型文件路径），则先检查该路径是否存在，
       存在则直接返回；否则抛出异常提示用户路径错误。
    2. 如果没有提供 custom_model_path，则按默认逻辑：
       - 调用 latest_logdir(root) 获取最新运行目录；
       - 在该目录中查找所有文件名包含 'model' 的模型文件；
       - 对模型文件按名称进行排序，取最后一个作为最新模型文件；
       - 返回完整模型文件路径。

    参数:
        root (str): 模型检查点根目录路径（例如 "hcanet-3.27_maddpg/data/logs/"）
        custom_model_path (str, optional): 用户自定义的模型文件完整路径；默认为 None

    返回:
        str: 模型文件的完整加载路径

    异常:
        如果自定义路径不存在或默认逻辑中找不到模型文件，则抛出 ValueError 异常。
    """
    # 如果用户指定了自定义模型路径，则直接使用该路径
    if custom_model_path is not None:
        if not os.path.exists(custom_model_path):
            raise ValueError("自定义模型路径不存在: " + custom_model_path)
        print(f'Using custom model path: {custom_model_path}')
        return custom_model_path

    # 默认逻辑：在指定的 root 目录中查找最新的运行目录
    try:
        last_run = latest_logdir(root)
    except Exception as e:
        raise ValueError("无法找到最新运行目录: " + str(e))
    
    # 在最新运行目录中查找所有包含 'model' 字符串的文件（可根据需要扩展为支持其他后缀）
    models = [file for file in os.listdir(last_run) if 'model' in file]
    if not models:
        raise ValueError("在目录 " + last_run + " 中未找到任何模型文件。")
    
    # 按名称排序。这里采用格式化字符串确保排序按数值大小进行（假设文件名类似 "model_XXXX.pt"）
    models.sort(key=lambda m: '{0:0>15}'.format(m))
    model = models[-1]
    print(f'Found latest model file: {model}')
    load_path = os.path.join(last_run, model)
    return load_path

def action_normalize(action):
    """
    规范化动作向量，使得其为单位向量（L2 范数为1）

    参数:
        action (torch.Tensor): 输入的动作张量

    返回:
        torch.Tensor: 归一化后的动作张量

    说明：
        归一化公式为：action / sqrt(sum(action^2))
    """
    # 计算动作的 L2 范数
    norm = torch.sqrt((action ** 2).sum())
    # 避免除以零的情况
    if norm.item() == 0:
        return action
    action = action / norm
    return action
