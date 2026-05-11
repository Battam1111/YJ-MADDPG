#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils.py 模块 - 模型加载和动作归一化工具函数
===================================================

本模块主要提供以下功能：
1. 获取最新检查点目录（latest_logdir）：遍历指定目录下的所有运行记录，并返回最新运行的完整路径。
2. 获取加载模型路径（get_load_path）：支持用户自定义模型路径加载，
   如果指定了 custom_model_path 参数：
     - 如果该路径为文件，则直接使用；
     - 如果该路径为目录，则在该目录中查找最新的模型文件（可通过 file_keyword 参数自定义过滤关键词，
       默认使用 "checkpoint"），对这些文件进行排序，选择排序后最后一个文件作为最新模型文件，并返回其完整路径。
   如果未指定 custom_model_path，则按照默认逻辑在最新运行目录中查找最新模型文件。
3. 动作归一化函数（action_normalize）：将输入动作张量归一化为单位向量（L2 范数为 1）。

作者：YourName
日期：202X-XX-XX
"""

import os
import torch

def latest_logdir(log_root):
    """
    获取日志目录中最新的运行目录

    参数:
        log_root (str): 日志根目录路径，例如 "hcanet-3.27_maddpg-MAAC/data/logs/"
    
    返回:
        str: 最新运行目录的完整路径

    异常:
        如果日志根目录中不存在任何子目录，则抛出 ValueError 异常。
    """
    if not os.path.exists(log_root):
        raise ValueError("日志根目录不存在: " + log_root)
    
    # 只考虑目录
    target_runs = [run for run in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, run))]
    if not target_runs:
        raise ValueError("日志目录中没有任何运行目录: " + log_root)
    
    # 假设目录名称包含日期时间信息，排序后最新的在最后
    target_runs.sort()
    latest_run = target_runs[-1]
    print(f'found the latest logdir: {latest_run}')
    return os.path.join(log_root, latest_run)

def get_load_path(root, custom_model_path="hcanet-3.27_maddpg-MAAC/data/logs/20250319-23:19:47", file_keyword="checkpoint"):
    """
    获取模型加载的完整路径。

    功能说明：
    1. 如果提供了 custom_model_path（用户自定义的模型路径），则先判断其是否存在：
       - 如果 custom_model_path 是一个文件，则直接返回该路径；
       - 如果 custom_model_path 是一个目录，则在该目录下查找所有文件名中包含 file_keyword（默认为 "checkpoint"）的文件，
         对这些文件进行排序，选择排序后最后一个文件作为最新的模型文件，并返回其完整路径。
    2. 如果没有提供 custom_model_path，则按照默认逻辑：
       - 调用 latest_logdir(root) 获取最新运行目录；
       - 在该目录中查找所有文件名中包含 file_keyword 的文件，并选择最新模型文件。

    参数:
        root (str): 模型检查点根目录路径（例如 "hcanet-3.27_maddpg-MAAC/data/logs/"）
        custom_model_path (str, optional): 用户自定义的模型路径，可以是文件也可以是目录，默认为指定的目录
        file_keyword (str, optional): 用于筛选模型文件的关键词，默认为 "checkpoint"

    返回:
        str: 模型文件的完整加载路径

    异常:
        如果自定义路径不存在或默认逻辑中找不到模型文件，则抛出 ValueError 异常。
    """
    # 如果用户指定了自定义模型路径，则先检查路径是否存在
    if custom_model_path is not None:
        if not os.path.exists(custom_model_path):
            raise ValueError("自定义模型路径不存在: " + custom_model_path)
        # 如果 custom_model_path 是一个文件，则直接返回该路径
        if os.path.isfile(custom_model_path):
            print(f'Using custom model file: {custom_model_path}')
            return custom_model_path
        # 如果 custom_model_path 是目录，则在该目录中查找包含 file_keyword 的文件
        elif os.path.isdir(custom_model_path):
            print(f'Using custom model directory: {custom_model_path}')
            models = [file for file in os.listdir(custom_model_path)
                      if file_keyword in file and os.path.isfile(os.path.join(custom_model_path, file))]
            if not models:
                raise ValueError("在自定义目录中未找到任何模型文件: " + custom_model_path)
            models.sort(key=lambda m: '{0:0>15}'.format(m))
            latest_model = models[-1]
            model_path = os.path.join(custom_model_path, latest_model)
            print(f'Found latest model file in custom directory: {latest_model}')
            return model_path
        else:
            raise ValueError("自定义模型路径不是有效的文件或目录: " + custom_model_path)

    # 若未提供 custom_model_path，则按默认逻辑在最新运行目录中查找模型文件
    try:
        last_run = latest_logdir(root)
    except Exception as e:
        raise ValueError("无法找到最新运行目录: " + str(e))
    
    models = [file for file in os.listdir(last_run)
              if file_keyword in file and os.path.isfile(os.path.join(last_run, file))]
    if not models:
        raise ValueError("在目录 " + last_run + " 中未找到任何模型文件。")
    
    models.sort(key=lambda m: '{0:0>15}'.format(m))
    latest_model = models[-1]
    print(f'Found latest model file: {latest_model}')
    load_path = os.path.join(last_run, latest_model)
    return load_path

def action_normalize(action):
    """
    规范化动作向量，使其成为单位向量（L2 范数为1）

    参数:
        action (torch.Tensor): 输入的动作张量

    返回:
        torch.Tensor: 归一化后的动作张量

    说明：
        归一化公式：action / sqrt(sum(action^2))
        若动作的 L2 范数为0，则直接返回原始动作，避免除零错误。
    """
    norm = torch.sqrt((action ** 2).sum())
    if norm.item() == 0:
        return action
    return action / norm
