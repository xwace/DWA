import copy

import numpy as np
import matplotlib.pyplot as plt
import math


class Info():
    def __init__(self):
        # define robot move speed ,accelerate,radius ...and so on
        # 定义机器人移动极限速度、加速度等信息
        self.v_min = -0.5
        self.v_max = 3.0
        self.w_max = 50.0 * math.pi / 180.0
        self.w_min = -50.0 * math.pi / 180.0
        self.vacc_max = 0.5
        self.wacc_max = 30.0 * math.pi / 180.0
        self.v_reso = 0.01
        self.w_reso = 0.1 * math.pi / 180.0
        self.radius = 1.0
        self.dt = 0.8#wxw
        # self.dt = 0.1
        self.predict_time = 4.0
        self.goal_factor = 1.0
        self.vel_factor = 1.0
        self.traj_factor = 1.0


# 定义机器人运动模型
# 返回坐标(x,y),偏移角theta,速度v,角速度w
def motion_model(x, u, dt):
    # robot motion model: x,y,theta,v,w
    x[0] += u[0] * dt * math.cos(x[2])
    x[1] += u[0] * dt * math.sin(x[2])
    x[2] += u[1] * dt
    x[3] = u[0]
    x[4] = u[1]
    return x


# 产生速度空间
def vw_generate(x, info):
    # generate v,w window for traj prediction
    Vinfo = [info.v_min, info.v_max,
             info.w_min, info.w_max]

    Vmove = [x[3] - info.vacc_max * info.dt,
             x[3] + info.vacc_max * info.dt,
             x[4] - info.wacc_max * info.dt,
             x[4] + info.wacc_max * info.dt]

    # 保证速度变化不超过info限制的范围
    vw = [max(Vinfo[0], Vmove[0]), min(Vinfo[1], Vmove[1]),
          max(Vinfo[2], Vmove[2]), min(Vinfo[3], Vmove[3])]

    return vw


# 依据当前位置及速度，预测轨迹
def traj_cauculate(x, u, info):
    ctraj = np.array(x)
    xnew = np.array(x)  # Caution!!! Don't use like this: xnew = x, it will change x value when run motion_modle below
    time = 0

    while time <= info.predict_time:  # preditc_time作用？循环40次
        xnew = motion_model(xnew, u, info.dt)
        ctraj = np.vstack((ctraj, xnew))
        time += info.dt#0.1

    return ctraj


def dwa_core(x, u, goal, info, obstacles):
    # the kernel of dwa
    vw = vw_generate(x, info)
    best_ctraj = np.array(x)
    min_score = 10000.0

    trajs = []

    # 速度v,w都被限制在速度空间里
    for v in np.arange(vw[0], vw[1], info.v_reso):
        for w in np.arange(vw[2], vw[3], info.w_reso):
            # cauculate traj for each given (v,w)
            ctraj = traj_cauculate(x, [v, w], info)
            # 计算评价函数
            goal_score = info.goal_factor * goal_evaluate(ctraj, goal)
            vel_score = info.vel_factor * velocity_evaluate(ctraj, info)
            traj_score = info.traj_factor * traj_evaluate(ctraj, obstacles, info)
            # 可行路径不止一条，通过评价函数确定最佳路径
            # 路径总分数 = 距离目标点 + 速度 + 障碍物
            # 分数越低，路径越优
            ctraj_score = goal_score + vel_score + traj_score
            # evaluate current traj (the score smaller,the traj better)
            if min_score >= ctraj_score:
                min_score = ctraj_score
                u = np.array([v, w])
                best_ctraj = ctraj
                trajs.append(ctraj)

    plt.ion()
    plt.plot(goal[0], goal[1], 'or', markersize=5)
    plt.plot([0, 14], [0, 0], '-k', linewidth=7)
    plt.plot([0, 14], [14, 14], '-k', linewidth=7)
    plt.plot([0, 0], [0, 14], '-k', linewidth=7)
    plt.plot([14, 14], [0, 14], '-k', linewidth=7)
    plt.plot([0, 6], [10, 10], '-y', linewidth=10)
    plt.plot([3, 8], [5, 5], '-y', linewidth=10)
    plt.plot([10, 10], [7, 13], '-y', linewidth=10)
    plt.plot(obstacles[:, 0], obstacles[:, 1], '*b', markersize=10)
    plt.plot(x[0], x[1], 'ob', markersize=5)
    plt.arrow(x[0], x[1], math.cos(x[2]), math.sin(x[2]), width=0.02, fc='red')
    plt.grid(True)

    for id, t in enumerate(trajs):
        if id %30 == 0:
            plt.plot(t[:, 0], t[:, 1], '-r', linewidth=1)

    plt.ioff()
    plt.show()

    return u, best_ctraj


# 距离目标点评价函数
def goal_evaluate(traj, goal):
    # cauculate current pose to goal with euclidean distance
    goal_score = math.sqrt((traj[-1, 0] - goal[0]) ** 2 + (traj[-1, 1] - goal[1]) ** 2)
    return goal_score


# 速度评价函数
def velocity_evaluate(traj, info):
    # cauculate current velocty score
    vel_score = info.v_max - traj[-1, 3]
    return vel_score


# 轨迹距离障碍物的评价函数
def traj_evaluate(traj, obstacles, info):
    # evaluate current traj with the min distance to obstacles
    min_dis = float("Inf")
    for i in range(len(traj)):
        for ii in range(len(obstacles)):
            current_dist = math.sqrt((traj[i, 0] - obstacles[ii, 0]) ** 2 + (traj[i, 1] - obstacles[ii, 1]) ** 2)

            if current_dist <= info.radius:
                return float("Inf")

            if min_dis >= current_dist:
                min_dis = current_dist

    return 1 / min_dis


# 生成包含障碍物的地图
def obstacles_generate():
    #	Map shape and obstacles:
    #	 ___________________________________
    #	|                                   |
    #	|                                   |
    #	|____________             |         |
    #	|                         |         |
    #	|                   goal  |         |
    #	|                   O     |         |
    #	|                         |         |
    #	|                                   |
    #	|       _____________               |
    #	|                                   |
    #	|     *(start)                      |
    #	|                                   |
    #	|___________________________________|

    obstacles = np.array([[0, 10],
                          [2, 10],
                          [4, 10],
                          [6, 10],
                          [3, 5],
                          [4, 5],
                          [5, 5],
                          [6, 5],
                          [7, 5],
                          [8, 5],
                          [10, 7],
                          [10, 9],
                          [10, 11],
                          [10, 13]])
    return obstacles

def local_traj_display(x, goal, current_traj, obstacles):
    # display current pose ,traj prodicted,map,goal
    plt.cla()
    plt.plot(goal[0], goal[1], 'or', markersize=1)
    plt.plot([0, 14], [0, 0], '-k', linewidth=7)
    plt.plot([0, 14], [14, 14], '-k', linewidth=7)
    plt.plot([0, 0], [0, 14], '-k', linewidth=7)
    plt.plot([14, 14], [0, 14], '-k', linewidth=7)
    plt.plot([0, 6], [10, 10], '-y', linewidth=10)
    plt.plot([3, 8], [5, 5], '-y', linewidth=10)
    plt.plot([10, 10], [7, 13], '-y', linewidth=10)
    plt.plot(obstacles[:, 0], obstacles[:, 1], '*b', markersize=8)
    plt.plot(x[0], x[1], 'ob', markersize=10)
    plt.arrow(x[0], x[1], math.cos(x[2]), math.sin(x[2]), width=0.02, fc='red')
    plt.plot(current_traj[:, 0], current_traj[:, 1], '-g', linewidth=2)
    plt.grid(True)
    plt.pause(0.001)

def main():
    x = np.array([2, 2, 45 * math.pi / 180, 0, 0])
    u = np.array([0, 0])
    goal = np.array([12, 4])
    info = Info()
    obstacles = obstacles_generate()
    global_traj = np.array(x)
    # plt.figure('DWA Algorithm')

    while 1:
        u, current_traj = dwa_core(x, u, goal, info, obstacles)
        x = motion_model(x, u, info.dt)
        global_traj = np.vstack((global_traj, x))
        # local_traj_display(x, goal, current_traj, obstacles)
        if math.sqrt((x[0] - goal[0]) ** 2 + (x[1] - goal[1]) ** 2) <= info.radius:
            print("Goal Arrived!")
            break

    # plt.plot(global_traj[:, 0], global_traj[:, 1], '-r')
    # plt.show()


if __name__ == "__main__":
    main()
