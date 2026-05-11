from time import sleep
import pybullet as p
import pybullet_data
from yaml import load, Loader
import random

from env.utils import *
from env.ou_noise import *
# from utils import * 
# from ou_noise import *

class Scence(object):
    def __init__(self, physicsClientId : int = 0):
        self._physics_client_id = physicsClientId
        param_path = "hcanet-3.27_maddpg/config/task.yaml"
        param_dict = load(open(param_path, "r", encoding="utf-8"), Loader=Loader)
        for key, value in param_dict.items():
            setattr(self, key, value)
        self.is_built = False # 是否已经调用过construct函数
        self.load_items = {} # 所有载入的entity的id
        self.signalPointId2data = {} # 所有信号点的数据量
        self.chargerId2state = {}# 指示是否被无人机占用（0 / 1）

    def construct(self):
        if self.is_built:       # 该函数只能执行一次
            raise Exception(f"plane_static_obstacle has been built!")
        self.is_built = True   

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # 添加平面
        self.load_items["plane"] = p.loadURDF("plane.urdf", useMaximalCoordinates=self.USE_MAX_COOR, physicsClientId=self._physics_client_id)
        # 添加围墙
        self.load_items["fence"] = addFence(
            center_pos=self.CENTER_POS,
            internal_length=self.INTERNAL_LENGTH,
            internal_width=self.INTERNAL_WIDTH,
            height=self.HEIGHT,
            thickness=self.THICKNESS,
            mass=self.FENCE_MASS,
            rgba=self.FENCE_COLOR,
            physicsClientId=self._physics_client_id
        )
        # 添加障碍物
        obstacle1 = addBox(
            pos=[self.CENTER_POS[0] + (self.INTERNAL_WIDTH / 6.), self.CENTER_POS[1] + (self.INTERNAL_LENGTH / 6.), self.CENTER_POS[2]],
            # halfExtents=[1.5, 1., 1. / 7. * self.HEIGHT],
            halfExtents=[0.5, 0.33, 1. / 7. * self.HEIGHT],
            physicsClientId=self._physics_client_id
        )
        obstacle2 = addBox(
            pos=[self.CENTER_POS[0] - (self.INTERNAL_WIDTH / 4.), self.CENTER_POS[1] - (self.INTERNAL_LENGTH / 6.), self.CENTER_POS[2]],
            # halfExtents=[3., 1., 1.1 / 7. * self.HEIGHT],
            halfExtents=[1., 0.33, 1.1 / 7. * self.HEIGHT],
            physicsClientId=self._physics_client_id
        )
        # obstacle3 = addBox(
        #     pos=[self.CENTER_POS[0] - (self.INTERNAL_WIDTH / 3.), self.CENTER_POS[1] + (self.INTERNAL_LENGTH / 7.), self.CENTER_POS[2]],
        #     # halfExtents=[1., 2.5, 1.3 / 7. * self.HEIGHT],
        #     halfExtents=[0.33, 0.83, 1.3 / 7. * self.HEIGHT],
        #     physicsClientId=self._physics_client_id
        # )
        obstacle4 = addBox(
            pos=[self.CENTER_POS[0] - (self.INTERNAL_WIDTH / 7. + 0.4), self.CENTER_POS[1] + (self.INTERNAL_LENGTH / 3. - 2.3), self.CENTER_POS[2]],
            # halfExtents=[2., 1., 1.5 / 7. * self.HEIGHT],
            halfExtents=[0.66, 0.33, 1.5 / 7. * self.HEIGHT],
            physicsClientId=self._physics_client_id
        )
        obstacle5 = addBox(
            pos=[self.CENTER_POS[0] + (self.INTERNAL_WIDTH / 3.), self.CENTER_POS[1] - (self.INTERNAL_LENGTH / 7.), self.CENTER_POS[2]],
            # halfExtents=[1., 2.5, 1.3 / 7. * self.HEIGHT],
            halfExtents=[0.33, 0.8, 1.3 / 7. * self.HEIGHT],
            physicsClientId=self._physics_client_id
        )
        # obstacle6 = addBox(
        #     pos=[self.CENTER_POS[0] + (self.INTERNAL_WIDTH / 7. + 0.4), self.CENTER_POS[1] - (self.INTERNAL_LENGTH / 3. - 2.3), self.CENTER_POS[2]],
        #     # halfExtents=[2., 1., 1.5 / 7. * self.HEIGHT],
        #     halfExtents=[0.66, 0.33, 1.5 / 7. * self.HEIGHT],
        #     physicsClientId=self._physics_client_id
        # )
        self.load_items["obstacle"] = [obstacle1, obstacle2, obstacle4, obstacle5]
        # 计算障碍物最大高度
        self.max_obstacleHeight = 0
        for obstacle in self.load_items["obstacle"]:
            self.max_obstacleHeight = max(self.max_obstacleHeight, p.getCollisionShapeData(obstacle, -1)[0][3][2])

        # 添加充电桩
        # charger1 = addBox(
        #     pos=[self.CENTER_POS[0] - (self.INTERNAL_WIDTH / 14. * 3. - 1.), self.CENTER_POS[1] - (self.INTERNAL_LENGTH /10. * 3. - 1.), self.CENTER_POS[2]],
        #     halfExtents=[0.5, 0.5, 1. / 4. * self.HEIGHT],
        #     rgba=self.CHARGER_COLOR,
        #     physicsClientId=self._physics_client_id
        # )
        # charger2 = addBox(
        #     pos=[self.CENTER_POS[0] + (self.INTERNAL_WIDTH / 14. * 3. - 1.), self.CENTER_POS[1] + (self.INTERNAL_LENGTH /10. * 3. - 1.), self.CENTER_POS[2]],
        #     halfExtents=[0.5, 0.5, 1. / 4. * self.HEIGHT],
        #     rgba=self.CHARGER_COLOR,
        #     physicsClientId=self._physics_client_id
        # )
        # self.load_items["charger"] = [charger1, charger2]
        # for chargerId in self.load_items["charger"]:
        #     self.chargerId2state[chargerId] = 0
        # 添加信号点
        # 各信号点位置、数据量根据随机数种子生成
        #print("###### start generate signalPoint ######")
        random.seed(self.RANDOM_SEED)
        self.load_items["signalPoint"] = []
        for i in range(self.NUM_SIGNAL_POINT):
            # 随机生成位置
            while True:
                signalPoint_pos = [self.CENTER_POS[0] + (self.INTERNAL_WIDTH / 2 * random.uniform(-1,1)), self.CENTER_POS[1] + (self.INTERNAL_LENGTH / 2 * random.uniform(-1,1)), self.CENTER_POS[2]]
                collide = False
                # 检查信号点是否与[障碍物、充电桩、其他信号点]产生碰撞
                for id in (self.load_items["fence"] + self.load_items["obstacle"]):
                    extents_current = p.getCollisionShapeData(id, -1)[0][3]
                    pos_current = p.getBasePositionAndOrientation(id)[0]
                    if ((abs(pos_current[0] - signalPoint_pos[0]) >= (extents_current[0] + self.SIGNAL_POINT_RADIUS)) or ((abs(pos_current[1] - signalPoint_pos[1]) >= (extents_current[1] + self.SIGNAL_POINT_RADIUS)))):
                        pass
                    else:
                        collide = True
                        break
                if collide == True:
                    pass
                else:
                    break
            signalPoint_id = addSphere(
                signalPoint_pos,
                radius=self.SIGNAL_POINT_RADIUS,
                rgba=self.SIGNAL_POINT_COLOR,
                physicsClientId=self._physics_client_id
                )
            self.load_items["signalPoint"].append(signalPoint_id)
        # 随机生成数据量
        # data_signalPoint = np.random.uniform(0, 1, self.NUM_SIGNAL_POINT)
        data_signalPoint = load_origData("hcanet-3.27_maddpg/env/data_signalPoint.npy")
        data_signalPoint_noise = OUNoise(self.NUM_SIGNAL_POINT, scale=0.5 / self.NUM_SIGNAL_POINT).noise().numpy()
        data_signalPoint += data_signalPoint_noise
        self.data_total = sum(data_signalPoint)
        for i in range(len(self.load_items["signalPoint"])):
            self.signalPointId2data[self.load_items["signalPoint"][i]] = data_signalPoint[i]
        #print("###### signalPoint generation done! ######")

if __name__ == "__main__":
    cid = p.connect(p.GUI)
    scence = Scence()
    scence.construct()
    # print('0', (p.getCollisionShapeData(scence.load_items["obstacle"][0], -1)[0][3]))
    # print('1', (p.getCollisionShapeData(scence.load_items["obstacle"][1], -1)[0][3]))
    # print('2', (p.getCollisionShapeData(scence.load_items["obstacle"][2], -1)[0][3]))
    # print('3', (p.getCollisionShapeData(scence.load_items["obstacle"][3], -1)[0][3]))
    # print('4', (p.getCollisionShapeData(scence.load_items["obstacle"][4], -1)[0][3]))
    # print('5', (p.getCollisionShapeData(scence.load_items["obstacle"][5], -1)[0][3]))
    # print('5', (p.getBasePositionAndOrientation(scence.load_items["obstacle"][5])[0]))
    print(scence.load_items)
    print(scence.signalPointId2data)
    btn_id = p.addUserDebugParameter("reset", 1, 0, 0)
    previous = p.readUserDebugParameter(btn_id)
    while True:
        if previous != p.readUserDebugParameter(btn_id):
            p.resetSimulation()
            previous = p.readUserDebugParameter(btn_id)

    p.disconnect(cid)