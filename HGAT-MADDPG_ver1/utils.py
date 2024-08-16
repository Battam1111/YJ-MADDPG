import os
import torch

def latest_logdir(log_root):
    target_runs = []
    for run in os.listdir(log_root):
         target_runs.append(run)
    target_runs.sort()
    latest_run = target_runs[-1]
    print(f'found the latest logdir: {latest_run}')
    return os.path.join(log_root, latest_run)

def get_load_path(root):
    try:
        last_run = latest_logdir(root)
    except:
        raise ValueError("No runs in this directory: " + root)

    load_run = last_run

    models = [file for file in os.listdir(load_run) if 'model' in file]
    models.sort(key=lambda m: '{0:0>15}'.format(m))
    model = models[-1]
    # model = "model_2000.pt"
    # print(f'found the latest model: {model}')
    load_path = os.path.join(load_run, model)
    return load_path

def action_normalize(action):
    # action = action / torch.sqrt((action**2).sum()) * torch.sqrt(torch.tensor(3))
    #action = action * torch.sqrt((action**2).sum()) / torch.sqrt(torch.tensor(3 * action.size()[0]))
    action = action / torch.sqrt((action**2).sum())
    return action