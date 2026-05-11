import pybullet as p
from yaml import load, Loader
import os
from collections import Counter
import numpy as np
from math import tan, sin, cos, sqrt, acos, pi
import functools
from functools import cmp_to_key

def addBox(pos: list, halfExtents: list, mass : float = 10000., rgba=[1., 1., 1., 1.], physicsClientId : int = 0):
    '''
    :pos: 位置
    :halfExtents: 三维方向的半径
    :mass: 质量
    :rgba: color
    '''
    visual_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=halfExtents,
        rgbaColor=rgba,
        physicsClientId=physicsClientId
        )
    collision_shape = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=halfExtents,
        physicsClientId=physicsClientId
        )
    entity_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[pos[0], pos[1], pos[2] + halfExtents[2]],
        physicsClientId=physicsClientId
    )
    return entity_id

def addFence(center_pos: list, internal_length: float, internal_width: float, height: float, thickness: float, mass: float = 10000., rgba: list = [1., 1., 1., 1.], physicsClientId : int = 0):
    """
    添加围墙
    :param center_pos:      围墙中心的坐标
    :param internal_length: 内部长
    :param internal_width:  内部宽
    :param thickness:       厚度
    :param mass:            质量
    :param rgba:            color
    :return                 四个id，代表组成围墙的四个box的id
    """
    # L1和L2代表长那条线面对面的两面墙，长度为internal_length + 2 * thickness
    L1 = addBox(
        pos=[center_pos[0] + internal_width / 2. + thickness / 2., center_pos[1], center_pos[2]],
        halfExtents=[thickness / 2., internal_length / 2. + thickness, height / 2.],
        mass=mass / 4.,
        rgba=rgba,
        physicsClientId=physicsClientId
    )
    L2 = addBox(
        pos=[center_pos[0] - internal_width / 2. - thickness / 2., center_pos[1], center_pos[2]],
        halfExtents=[thickness / 2., internal_length / 2. + thickness, height / 2.],
        mass=mass / 4.,
        rgba=rgba,
        physicsClientId=physicsClientId
    )
    # W1和W2代表宽那条线面对面的两面墙，长度为internal_length + 2 * thickness
    W1 = addBox(
        pos=[center_pos[0], center_pos[1] + internal_length / 2. + thickness / 2., center_pos[2]],
        halfExtents=[internal_width / 2., thickness / 2., height / 2.],
        mass=mass / 4.,
        rgba=rgba,
        physicsClientId=physicsClientId
    )
    W2 = addBox(
        pos=[center_pos[0], center_pos[1] - internal_length / 2. - thickness / 2., center_pos[2]],
        halfExtents=[internal_width / 2., thickness / 2., height / 2.],
        mass=mass / 4.,
        rgba=rgba,
        physicsClientId=physicsClientId
    )
    return [L1, L2, W1, W2]

def addSphere(pos : list, radius : float, mass : float = 10000., rgba : list = [1., 1. ,1., 1.], physicsClientId : int = 0):
    '''
    添加球
    '''
    visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=physicsClientId)
    collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=physicsClientId)
    entity_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[pos[0], pos[1], pos[2] + radius],
        physicsClientId=physicsClientId
    )
    return entity_id

def rayTest(robot_id : int, ray_length : float, ray_num_horizontal : int = 5, ray_num_vertical : int = 4, base_radius : float = 0.25, physicsClientId : int = 0):
    basePos, baseOrientation = p.getBasePositionAndOrientation(robot_id, physicsClientId=physicsClientId)
    matrix = p.getMatrixFromQuaternion(baseOrientation, physicsClientId=physicsClientId)
    basePos = np.array(basePos)
    matrix = np.array(matrix).reshape([3, 3])
    # 选定在机器人的本地坐标系中中心到几个激光发射点的向量
    # 此处的逻辑为先计算出local坐标系中的距离单位向量，再变换到世界坐标系中
    #unitRayVecs = np.array([[cos(alpha), sin(alpha), 0]  for alpha in np.linspace(-np.pi / 2., np.pi / 2., ray_num_horizontal)])
    unitRayVecs = np.array([[cos(alpha), sin(alpha), 0]  for alpha in np.linspace(-np.pi, np.pi, ray_num_horizontal)])
    #unitRayVecs = np.append(unitRayVecs, np.array([[0, cos(alpha), sin(alpha)]  for alpha in np.linspace(-np.pi / 2., -np.pi / 6., ray_num_vertical)])).reshape(-1, 3)
    unitRayVecs = np.append(unitRayVecs, np.array([[0, sin(alpha), cos(alpha)]  for alpha in np.linspace(np.pi / 2., 3. * np.pi / 2., ray_num_vertical)])).reshape(-1, 3)
    unitRayVecs = np.append(unitRayVecs, np.array([[sin(alpha), 0, cos(alpha)]  for alpha in np.linspace(np.pi / 2., 3. * np.pi / 2., ray_num_vertical)])).reshape(-1, 3)
    unitRayVecs = np.append(unitRayVecs, np.array([[0, cos(alpha), sin(alpha)]  for alpha in np.linspace(0., np.pi, ray_num_vertical)])).reshape(-1, 3)
    unitRayVecs = np.append(unitRayVecs, np.array([[cos(alpha), 0, sin(alpha)]  for alpha in np.linspace(0., np.pi, ray_num_vertical)])).reshape(-1, 3)
    unitRayVecs = unitRayVecs.dot(matrix.T)
    # 通过广播运算得到世界坐标系中所有激光发射点的坐标
    rayBegins = basePos + base_radius * unitRayVecs
    rayTos = rayBegins + ray_length * unitRayVecs
    results = p.rayTestBatch(rayBegins, rayTos, physicsClientId=physicsClientId)
    return unitRayVecs, rayBegins, rayTos, results

# def check_obstacle(robot_pos, robot_orientation, angle, traget_SP_pos, base_radius : float = 0.25, physicsClientId : int = 0):
#     matrix = p.getMatrixFromQuaternion(robot_pos, robot_orientation)
#     basePos = np.array(robot_pos)
#     matrix = np.array(matrix).reshape([3, 3])
#     unitRayVecs = np.array([angle])
#     unitRayVecs = unitRayVecs.dot(matrix.T)
#     rayBegins = basePos + base_radius * unitRayVecs
#     rayTos = rayBegins + caculate_distance(robot_pos, traget_SP_pos) * unitRayVecs
#     results = p.rayTestBatch(rayBegins, rayTos, physicsClientId=physicsClientId)
#     return unitRayVecs, rayBegins, rayTos, results

class SegmentsIntersect(object):
    def __init__(self, p1, p2, q1, q2):
        self.result = self.judge_segments_intersect(p1, p2, q1, q2)

    def __sort_by_coordiante(self, x1, x2, k):
        if x1[k] < x2[k]:
            return -1
        elif x1[k] == x2[k]:
            return 0
        else:
            return 1

    def judge_segments_intersect(self, p1, p2, q1, q2):
        p = self.minus(p2, p1)
        q = self.minus(q2, q1)

        denominator = self.crossmultiply(p, q)  # p × q
        t_molecule = self.crossmultiply(self.minus(q1, p1), q)  # (q1 - p1) × q

        if denominator == 0:
            if t_molecule == 0:  # 分子分母都为零时，共线
                p_q = [p1, p2, q1, q2]
                if p1 != q1 and p1 != q2 and p2 != q1 and p2 != q2:
                    p_q = sorted(p_q, key=cmp_to_key(functools.partial(self.__sort_by_coordiante, k = 1 if (p2[0]-p1[0])/(p2[1]-p1[1]) == 0 else 0))) # 当线段平行于y轴时，需要用y坐标来排序
                    if p_q[0:2] == [p1, p2] or p_q[0:2] == [p2, p1] or p_q[0:2] == [q1, q2] or p_q[0:2] == [q2, q1]:  # 共线+没有交点的情况
                        return "collinear separation"
                    else:  # 共线+有重合的情况
                        return "collinear part coincide"
                else:  # 共线+端点重合，可以继续细分为两对端点都重合（相同线段）or只有一对端点重合 这两种情况
                    return "collinear part coincide"
            else:  # 分母为零，分子不为零，平行
                return "parallel"

        t = t_molecule / denominator
        if t >= 0 and t <= 1:
            u_molecule = self.crossmultiply(self.minus(q1, p1), p)  # (q1 - p1) × p
            u = u_molecule / denominator
            if u >= 0 and u <= 1:  # t, u都满足[0,1]区间，返回交点坐标
                return self.plus(p1, self.nummultiply(t, p))
            else:  # u超出区间，则相离
                return "separation"
        else:  # t超出区间，则相离
            return "separation"

    #向量相加
    def plus(self, a, b):
        c = []
        for i, j in zip(a, b):
            c.append(i+j)
        return c

    #向量相减
    def minus(self, a, b):
        c = []
        for i, j in zip(a, b):
            c.append(i-j)
        return c

    #向量叉乘
    def crossmultiply(self, a, b):
        return a[0]*b[1]-a[1]*b[0]

    #向量数乘
    def nummultiply(self, x, a):
        c = []
        for i in a:
            c.append(x*i)
        return c

def exist_obstacle(robot_pos, closest_SP_pos, obstacle_id_list, physicsClientId=0):
    for obstacle_id in obstacle_id_list:
        obstacle_pos, _ = p.getBasePositionAndOrientation(obstacle_id, physicsClientId=physicsClientId)
        obstacle_extent = list(p.getCollisionShapeData(obstacle_id, -1)[0][3])
        p1 = [obstacle_pos[0] - obstacle_extent[0], obstacle_pos[1] - obstacle_extent[1]]
        p2 = [obstacle_pos[0] + obstacle_extent[0], obstacle_pos[1] - obstacle_extent[1]]
        p3 = [obstacle_pos[0] + obstacle_extent[0], obstacle_pos[1] + obstacle_extent[1]]
        p4 = [obstacle_pos[0] - obstacle_extent[0], obstacle_pos[1] + obstacle_extent[1]]
        point_list = [p1, p2, p3, p4, p1]
        for i in range(4):
            if isinstance(SegmentsIntersect(robot_pos, closest_SP_pos, point_list[i], point_list[i + 1]).result, list):
                return True, obstacle_id
            else:
                continue

    return False, None

def set_forceDirection(keyBoards : bool = False, force : float = 100.):
    '''
    keyBoards
        1. True: 从键盘获取
        2. False: 使用模型输出
    '''
    if keyBoards:
        keys = p.getKeyboardEvents()
        forward, turn, up = 0, 0, 0
        for k, v in keys.items():
            if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_TRIGGERED)):
                turn = -1
            if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_RELEASED)):
                turn = 0
            if (k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_TRIGGERED)):
                turn = 1
            if (k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_RELEASED)):
                turn = 0

            if (k == p.B3G_UP_ARROW and (v & p.KEY_WAS_TRIGGERED)):
                forward = 1
            if (k == p.B3G_UP_ARROW and (v & p.KEY_WAS_RELEASED)):
                forward = 0
            if (k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_TRIGGERED)):
                forward = -1
            if (k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_RELEASED)):
                forward = 0

            if (k == ord('o') and (v & p.KEY_WAS_TRIGGERED)):
                up = 1
            if (k == ord('o') and (v & p.KEY_WAS_RELEASED)):
                up = 0
            if (k == ord('p') and (v & p.KEY_WAS_TRIGGERED)):
                up = -1
            if (k == ord('p') and (v & p.KEY_WAS_RELEASED)):
                up = 0
        force_list = [forward * force, turn * force, up * force]
        return force_list
    else:
        pass


def set_orientation(keyBoards : bool = False):
    '''
    keyBoards
        1. True: 从键盘获取
        2. False: 使用模型输出
    '''
    pass

def caculate_distance(PosA : list = None, PosB : list = None):
    return sqrt(sum((np.array(list(PosA)) - np.array(list(PosB)))**2))

def caculate_2D_distance(PosA : list = None, PosB : list = None):
    return sqrt(sum((np.array(list(PosA)[:2]) - np.array(list(PosB)[:2]))**2))

# def fairness(signalPointId2dataChanged: dict):
#     if sum(list(signalPointId2dataChanged.values())) >= 0.:
#         return 1.
#     else:
#         data_np = np.array(list(signalPointId2dataChanged.values()))
#         num = Counter(data_np < 0.)[True]
#         return (data_np.sum()**2) / (num * (data_np**2).sum())

def fairness(data_orig, data_final):
    data_orig, data_final = np.array(data_orig), np.array(data_final)
    # print(data_orig, data_final)
    num_SP = len(data_orig)
    diff_origAndFinal = data_orig - data_final
    norm_diff_origAndFinal = diff_origAndFinal / data_orig
    return sum(norm_diff_origAndFinal)**2 / (num_SP * sum(norm_diff_origAndFinal**2))

def velocilty_normalize(velocity):
    velocity = np.array(velocity)
    scale_velocity = np.sqrt((velocity**2).sum())
    if scale_velocity > 0.:
        velocity = velocity / np.sqrt((velocity**2).sum())
    else:
        pass
    return velocity

def direction_normalize(direction: np.ndarray):
    direction = direction / np.sqrt((direction**2).sum())
    return direction

def create_origData(save_path, num_signal_point, seed):
    np.random.seed(seed)
    data_signalPoint = np.random.uniform(0, 1, num_signal_point)
    np.save(save_path, data_signalPoint)

def load_origData(save_path):
    data_signalPoint = np.load(save_path, allow_pickle=True)
    return data_signalPoint

if __name__ == "__main__":
    param_dict = {}
    for file in os.listdir("hcanet-3.27_maddpg/config"):
        path = "hcanet-3.27_maddpg/config/" + file
        param_dict_current = load(open(path, "r", encoding="utf-8"), Loader=Loader)
        param_dict.update(param_dict_current)
    save_path = "hcanet-3.27_maddpg/env/data_signalPoint.npy"
    create_origData(save_path, param_dict["NUM_SIGNAL_POINT"], param_dict["RANDOM_SEED"])
    data_signalPoint = load_origData(save_path)
    print(data_signalPoint)