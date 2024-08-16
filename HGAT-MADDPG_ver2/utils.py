import os
import torch

def latest_logdir(log_root):
    """
    找到日志目录中最新的运行目录。
    
    参数:
    log_root (str): 日志根目录路径。
    
    返回:
    str: 最新的运行目录的路径。
    """
    target_runs = []
    
    # 遍历日志目录，收集所有运行目录
    for run in os.listdir(log_root):
        target_runs.append(run)
    
    # 按目录名称排序，找到最新的运行目录
    target_runs.sort()
    latest_run = target_runs[-1]
    print(f'found the latest logdir: {latest_run}')
    
    return os.path.join(log_root, latest_run)

def get_load_path(root):
    """
    获取最新运行目录中最新的模型文件路径。
    
    参数:
    root (str): 日志根目录路径。
    
    返回:
    str: 最新的模型文件路径。
    
    异常:
    ValueError: 当目录中没有任何运行时抛出。
    """
    try:
        last_run = latest_logdir(root)
    except IndexError:
        raise ValueError("No runs in this directory: " + root)
    
    # 获取最新运行目录中的所有模型文件
    models = [file for file in os.listdir(last_run) if 'model' in file]
    
    # 按模型文件名称排序，找到最新的模型文件
    models.sort(key=lambda m: '{0:0>15}'.format(m))
    model = models[-1]
    
    load_path = os.path.join(last_run, model)
    return load_path

def action_normalize(action):
    """
    规范化动作向量，使其单位化。
    
    参数:
    action (torch.Tensor): 输入的动作向量。
    
    返回:
    torch.Tensor: 规范化后的动作向量。
    """
    # 规范化动作向量，使其单位化
    action = action / torch.sqrt((action ** 2).sum())
    return action
