import numpy as np

import random

from sensor_env import SensorEnv

'''
1.初始化状态state,初始化q表
2.查找每个传感器范围内的目标
3.随机跟踪一个目标（1000m范围内），第一步随机动作
4.根据状态得到回报
5.更新Q表
6.利用Q表选择下一步动作
'''
#===========================================================
def epsilon_greedy_policy(Q,epsilon):
    seed = random.random()
    if epsilon>=seed:#如果epsilon大于随机数，利用
        action_index = np.argmax(Q)
    else:#如果epsilon小于随机数，探索
        action_index = random.randint(0,len(Q)-1)#0-127的随机整数
    return action_index
#===========================================================


s = SensorEnv()
action_space = s.get_state_start()
targets = s.targets
sensors = s.sensor
print('状态矩阵action_space为:')
print(action_space)
print('一共有'+str(len(action_space))+'个状态')
EPSILON = 0.2#贪婪策略参数
DISCOUNT_FACTOR = 0.6#折扣因子
ALPHA = 0.5#更新步长
N = 200
Q = np.zeros(len(action_space))##step 1：初始化Q矩阵
print('Q值矩阵的长度为'+str(len(Q)))
count = 10

#policy = epsilon_greedy_policy(Q, EPSILON, len(action_space))#策略是用易普希龙策略在Q表中选值
for i_episode in range(N):  #step 2：小于N时，步骤 3 到步骤 5 会一直被重复。
    for t in range(count):                             #小于最大步长10时
        action_index = epsilon_greedy_policy(Q,EPSILON)#step 3:选取下一个动作 a
        action = action_space[action_index]            #step 4：跟新当前的state
                                                       # step 4：计算reward
        dealt = np.zeros(len(targets))#尺寸为目标数的矩阵
        for i in range(len(targets)):#遍历每个目标
            index = i#这个索引应该是从0一直到目标数
            if index in action:   #如果在上面更新得到的state含有该目标，比如2
                dealt[i] = 1      #dealt=[0,0,1,0,0]
                                  #求得了列表dealt=[0,1,1,0,0]
        if np.sum(dealt) == len(targets):#所有的目标都被集中跟踪到，即每个传感器跟踪到了每个目标
            done = True
            r = +10                   #如果每一个目标都被跟踪到了，+10分，完成任务
        else:
            done = False              #如果有遗落的目标，-1分，未完成任务
            r = -1

        powerNeed = 0
                             # sensor和target的距离要足够接近
        for i in range(len(action)):#i从0到4
            target_id = action[i]
            sensor_id = sensors[i]
            distance,deviation = s.Observation_deviation(sensor_id,target_id)
            powerNeed += distance*0.01
        r = r + (powerNeed * -1)#根据耗电量修改回报

        best_next_action_index = np.argmax(Q)#选取Q值里最大的一个
        td_target = r + DISCOUNT_FACTOR * Q[best_next_action_index]
        td_delta = td_target - Q[action_index]
        Q[action_index] += ALPHA * td_delta#step 5：利用时序差分法更新Q值表

        if done:
            break

    bestActionInd = np.argmax(Q)#返回中分配方式中q值最大的一个
    print('第'+str(i_episode)+'代的最佳状态index为：'+str( bestActionInd))

print('最优化分为：')
print(action_space[bestActionInd])
print('Q值表为：')
print(Q)