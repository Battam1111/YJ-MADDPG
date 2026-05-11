from torch._C import import_ir_module
from yaml import load, Loader
import numpy as np
import json
import pybullet as p
import matplotlib.pyplot as plt
import time

from env.scene import Scence

if __name__ == "__main__":
    color_list = [
        [1, 0, 0], # 红
        # [0, 1, 0], # 绿
        [0.5, 0, 0.5], # 紫
        # [1, 1, 0], # 黄
        [1, 0.84, 0] # 土黄
    ]

    param_path = "./config/task.yaml"
    param_dict = load(open(param_path, "r", encoding="utf-8"), Loader=Loader)
    # 设置随机数种子
    np.random.seed(param_dict["RANDOM_SEED"])
    cid = p.connect(p.GUI)
    scence = Scence()
    scence.construct()
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(
        cameraDistance=9,
        cameraYaw=0,
        cameraPitch=-89,
        cameraTargetPosition=[0, 0, 0]
    )

    # file_path = "./data/eval/trajectory_max_totalDataCollected.txt"
    # file_path = "/Users/ff0kk/learn/群值感知/hk/mine2/AAAC_v0.1_3_i_mR_SF_RB_OH_UA2_UC1_PER_mi_BS_aSP_DD_v2_aSPA/data/eval/eval_perStep/9_0.9534063100233414_trajectory.txt"
    file_path = "./data/tra/trajectory_3300.json"
    with open(file_path, "r") as file:
        trajectory_data = file.read()
    trajectory_data = np.array((json.loads(trajectory_data)))

    for i in range(len(trajectory_data) - 1):
        for j in range(3):
            # pass
            p.addUserDebugLine(
                trajectory_data[i][j], trajectory_data[i+1][j], 
                lineColorRGB=color_list[j], lifeTime=30000000, lineWidth=5)
        # p.stepSimulation()

    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=[0, 0, 22],
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector=[0, 1, 0])
    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=45.0,
        aspect=1.0,
        nearVal=0.1,
        farVal=22.1)
    w, h, rgbPixels, depthPixels, segPixels = p.getCameraImage(
        1600, 1600,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix)
    rgbPixels = np.array(rgbPixels).reshape(w, h, 4)
    plt.imshow(rgbPixels)
    # plt.title("rgbPixels")
    plt.axis("off")
    plt.savefig("./test.pdf")

    print("done")
    time.sleep(20)