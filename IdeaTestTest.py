import bisect
import csv
import math
import random
import time
import networkx as nx


# from matplotlib import pyplot as plt

# 策略类
class Strategy:

    def __init__(self, theStrategy, StateTime, EnqueueTime, UsedTime=-1):
        self.theStrategy = theStrategy
        self.StateTime = StateTime
        self.UsedTime = UsedTime
        self.EnqueueTime = EnqueueTime

    def setUsedTime(self, inputTime):
        self.UsedTime = inputTime

    def getUsedTime(self):
        if self.UsedTime == -1:
            exit()
        return self.UsedTime

# 策略队列，循环队列实现
class StrategiesQueue:
    def __init__(self, size):
        self.size = size
        self.queue = [-1] * size
        self.front = self.rear = -1
        self.lastStrategies = -1

    def enqueue(self, item):
        if (self.rear + 1) % self.size == self.front:
            pass
        elif self.front == -1:
            self.front = self.rear = 0
            self.queue[self.rear] = item
        else:
            self.rear = (self.rear + 1) % self.size
            self.queue[self.rear] = item

    def dequeue(self, NowTime):
        if self.front == -1:
            return -1
        elif self.front == self.rear:
            item = self.queue[self.front]
            item.setUsedTime(NowTime)
            self.front = self.rear = -1
            self.lastStrategies = item
            return item
        else:
            item = self.queue[self.front]
            item.setUsedTime(NowTime)
            self.front = (self.front + 1) % self.size
            self.lastStrategies = item
            return item

    def display(self):
        if self.front == -1:
            print("队列为空")
        elif self.rear >= self.front:
            for i in range(self.front, self.rear + 1):
                print(self.queue[i].theStrategy, end=" ")
            print()
        else:
            for i in range(self.front, self.size):
                print(self.queue[i].theStrategy, end=" ")
            for i in range(0, self.rear + 1):
                print(self.queue[i].theStrategy, end=" ")
            print()

    def is_empty(self):
        return self.front == -1

    def peekFront(self):
        return self.queue[self.front]


IndividualNum = 3000  # 个体数量
AverageNum = 1  # 实验平均次数


AverageDegree = 6

# 图生成
G = nx.watts_strogatz_graph(IndividualNum, AverageDegree, 0.2)
# G = nx.watts_strogatz_graph(IndividualNum, 6, 0)
# G = nx.erdos_renyi_graph(IndividualNum,p = AverageDegree / (IndividualNum - 1))
# G = nx.barabasi_albert_graph(IndividualNum, 2)
# G = nx.random_regul2ar_graph(d=AverageDegree, n=IndividualNum)
# G = nx.powerlaw_cluster_graph(IndividualNum, 2 , 0.5)

# 个体不同速率更新
# LambdaList = [0.8/Lambda[1] for Lambda in nx.degree(G)]
# MuList = [1/Mu[1] for Mu in nx.degree(G)]

# # 个体相同速率更新
LambdaList = [0.5 for _ in range(IndividualNum)]
MuList = [float(1) for _ in range(IndividualNum)]
total_lambda = sum(LambdaList)

# 博弈参数
b = 1.2
c = 1
sigma = 0.1
R, S, T, P = 1, 0, b, 0


IteratedStep =  3000  # 迭代次数
IteratedTime = ((1/total_lambda) * IndividualNum * IteratedStep)     # 迭代时间


#____________________________预处理_______________________________
# 在初始化时预计算
nodes_list = list(G.nodes)
# 假设 LambdaList 与 nodes_list 顺序一一对应
cumulative_weights = []
total = 0
for w in LambdaList:
    total += w
    cumulative_weights.append(total)


# 获取个体收益
def get_payoff(G, node):
    payoff = 0

    for neighbor in list(G.neighbors(node)):

        # 当邻居或自身没有策略时，收益为0
        if (NowStrategiesList[node] == -1) or (NowStrategiesList[neighbor] == -1):
            payoff += sigma
            continue

        if NowStrategiesList[node].theStrategy == 1 and NowStrategiesList[neighbor].theStrategy == 1:
            payoff += R
        elif NowStrategiesList[node].theStrategy == 1 and NowStrategiesList[neighbor].theStrategy == 0:
            payoff += S
        elif NowStrategiesList[node].theStrategy == 0 and NowStrategiesList[neighbor].theStrategy == 1:
            payoff += T
        elif NowStrategiesList[node].theStrategy == 0 and NowStrategiesList[neighbor].theStrategy == 0:
            payoff += P

    return payoff


# 用于判断每个时刻策略队列中的策略是否弹出
def UpdateALL(G, NowTime):
    # StrategiesQueueList[10].display()
    for UpdateNode in G.nodes:
        if NowStrategiesList[UpdateNode] == -1:  # 如果此时个体没有策略

            if StrategiesQueueList[UpdateNode].is_empty():
                continue  # 直接走，是Loner

            # 先弹出一个最早的让他变成有策略的
            NowStrategiesList[UpdateNode] = StrategiesQueueList[UpdateNode].dequeue((StrategiesQueueList[UpdateNode].peekFront()).EnqueueTime)

            # 如果策略队列不空，就一直弹到用没过期的
            while (NowTime - NowStrategiesList[UpdateNode].StateTime >= NowStrategiesList[UpdateNode].UsedTime):

                if StrategiesQueueList[UpdateNode].is_empty():
                    NowStrategiesList[UpdateNode] = -1
                    break  # 过期了并且队列为空。令为-1，是Loner

                if (StrategiesQueueList[UpdateNode].lastStrategies).UsedTime +(StrategiesQueueList[UpdateNode].lastStrategies).StateTime < (StrategiesQueueList[UpdateNode].peekFront()).EnqueueTime:
                    NowStrategiesList[UpdateNode] = StrategiesQueueList[UpdateNode].dequeue((StrategiesQueueList[UpdateNode].peekFront()).EnqueueTime)
                else:
                    NowStrategiesList[UpdateNode] = StrategiesQueueList[UpdateNode].dequeue((StrategiesQueueList[UpdateNode].lastStrategies).UsedTime + (StrategiesQueueList[UpdateNode].lastStrategies).StateTime)


                # 检查是否把队列弹空了
                if NowStrategiesList[UpdateNode] == -1:
                    break


        else:  # 否则判断当前策略是否超时，超时就让新的策略顶上来
            # 一直弹到用没过期的
            while (NowTime - NowStrategiesList[UpdateNode].StateTime >= NowStrategiesList[UpdateNode].UsedTime):
                NowStrategiesList[UpdateNode] = StrategiesQueueList[UpdateNode].dequeue((StrategiesQueueList[UpdateNode].lastStrategies).UsedTime +(StrategiesQueueList[UpdateNode].lastStrategies).StateTime)
                # 检查是否把队列弹空了
                if NowStrategiesList[UpdateNode] == -1:
                    break
    # StrategiesQueueList[10].display()
    # print("________")


# 用于判断每个时刻策略队列中的策略是否弹出
def Update(G, NowTime, UpNode):

    UP_nodes = set(G.neighbors(UpNode))  # 获取UpdateNode的邻居，并转化为set以去重
    for node in UP_nodes.copy():  # 用copy避免在迭代时修改集合
        UP_nodes.update(G.neighbors(node))  # 将node的邻居添加到UP_nodes中
    UP_nodes = list(UP_nodes)

    for UpdateNode in UP_nodes:
        if NowStrategiesList[UpdateNode] == -1:  # 如果此时个体没有策略
            if StrategiesQueueList[UpdateNode].is_empty():
                continue  # 直接走，是Loner

            # 先弹出一个最早的让他变成有策略的
            NowStrategiesList[UpdateNode] = StrategiesQueueList[UpdateNode].dequeue(
                (StrategiesQueueList[UpdateNode].peekFront()).EnqueueTime)

            # 如果策略队列不空，就一直弹到用没过期的
            while (NowTime - NowStrategiesList[UpdateNode].StateTime >= NowStrategiesList[UpdateNode].UsedTime):
                if StrategiesQueueList[UpdateNode].is_empty():
                    NowStrategiesList[UpdateNode] = -1
                    break  # 过期了并且队列为空。令为-1，是Loner

                if (StrategiesQueueList[UpdateNode].lastStrategies).UsedTime + \
                   (StrategiesQueueList[UpdateNode].lastStrategies).StateTime < \
                   (StrategiesQueueList[UpdateNode].peekFront()).EnqueueTime:
                    NowStrategiesList[UpdateNode] = StrategiesQueueList[UpdateNode].dequeue(
                        (StrategiesQueueList[UpdateNode].peekFront()).EnqueueTime)
                else:
                    NowStrategiesList[UpdateNode] = StrategiesQueueList[UpdateNode].dequeue(
                        (StrategiesQueueList[UpdateNode].lastStrategies).UsedTime +
                        (StrategiesQueueList[UpdateNode].lastStrategies).StateTime)

                # 检查是否把队列弹空了
                if NowStrategiesList[UpdateNode] == -1:
                    break

        else:  # 否则判断当前策略是否超时，超时就让新的策略顶上来
            # 一直弹到用没过期的
            while (NowTime - NowStrategiesList[UpdateNode].StateTime >= NowStrategiesList[UpdateNode].UsedTime):
                NowStrategiesList[UpdateNode] = StrategiesQueueList[UpdateNode].dequeue(
                    (StrategiesQueueList[UpdateNode].lastStrategies).UsedTime +
                    (StrategiesQueueList[UpdateNode].lastStrategies).StateTime)
                # 检查是否把队列弹空了
                if NowStrategiesList[UpdateNode] == -1:
                    break



# 模仿策略
def ImitateUpdate(G, UpdateNode, NowTime):

    # 剔除无策略邻居
    SelectNeighbors = [n for n in G[UpdateNode] if NowStrategiesList[n] != -1]

    # 如果邻居都没有策略
    if len(SelectNeighbors) == 0:
        # 自己没有策略那么用先验
        if NowStrategiesList[UpdateNode] == -1:
            StrategiesQueueList[UpdateNode].enqueue(
                Strategy((StrategiesQueueList[UpdateNode].lastStrategies).theStrategy, random.expovariate(MuList[UpdateNode]) , EnqueueTime=NowTime))
            return
        # 自己有策略就用自己的
        else:
            StrategiesQueueList[UpdateNode].enqueue(
                Strategy(NowStrategiesList[UpdateNode].theStrategy, random.expovariate(MuList[UpdateNode]) , EnqueueTime=NowTime))
            return

    # 随机挑选一个邻居
    Neighbor = random.choice(SelectNeighbors)

    NeighborPayoff = get_payoff(G, Neighbor)
    SelfPayoff = get_payoff(G, UpdateNode)

    Kd = max(G.degree[Neighbor], G.degree[UpdateNode])

    if NeighborPayoff > SelfPayoff:
        probability = (NeighborPayoff - SelfPayoff) / (b * Kd)
    else:
        probability = 0

    if random.random() <= probability:
        StrategiesQueueList[UpdateNode].enqueue(
            Strategy(NowStrategiesList[Neighbor].theStrategy, random.expovariate(MuList[UpdateNode]) , EnqueueTime=NowTime))
        return
    else:  # 不模仿
        if NowStrategiesList[UpdateNode] == -1:
            StrategiesQueueList[UpdateNode].enqueue(
                Strategy((StrategiesQueueList[UpdateNode].lastStrategies).theStrategy, random.expovariate(MuList[UpdateNode]) , EnqueueTime=NowTime))
            return
        else:
            StrategiesQueueList[UpdateNode].enqueue(
                Strategy(NowStrategiesList[UpdateNode].theStrategy, random.expovariate(MuList[UpdateNode]) , EnqueueTime=NowTime))
            return




# 初始化策略队列

def StrategiesQueueListInit(StrategiesQueueList, NowTime):
    total_queues = len(StrategiesQueueList)
    half = total_queues // 2

    # 随机选择一半的队列添加 0
    selected_indices = random.sample(range(total_queues), half)
    for index in selected_indices:
        StrategiesQueueList[index].enqueue(Strategy(0, random.expovariate(MuList[index]) , EnqueueTime=NowTime))

    # 向剩余的队列添加 1
    for index, queue in enumerate(StrategiesQueueList):
        if index not in selected_indices:
            queue.enqueue(Strategy(1, random.expovariate(MuList[index]) , EnqueueTime=NowTime))

    return StrategiesQueueList


def GetUpTime():

    # 生成下一次的更新时间
    uptime = random.expovariate(total_lambda)

    return uptime


def GetUpNode():
    r = random.uniform(0, total)
    idx = bisect.bisect(cumulative_weights, r)
    return nodes_list[idx]

# MAINLOOP
for i in range(AverageNum):

    result_C = []  # 结果
    result_D = []  # 结果
    result_L = []  # 结果
    times = []

    # print(j)
    # 初始策略集合，随机生成一个合作者，合作策略持续1个time step
    NowStrategiesList = [-1 for _ in range(IndividualNum)]

    NowTime = 0  # 初始化时间

    StrategiesQueueList = [StrategiesQueue(300) for _ in range(IndividualNum)]  # 策略队列声明，队列长度尽可能大防止上溢
    StrategiesQueueList = StrategiesQueueListInit(StrategiesQueueList, NowTime)  # 初始化队列

    UpdateALL(G, NowTime)  # 弹出初始策略


    last_recorded_time = -1

    # 用于记录各个步骤的运行时间
    times_GetUpTime = []
    times_GetUpNode = []
    times_Update = []
    times_ImitateUpdate = []

    iteration_count = 0

    while NowTime < IteratedTime:
        # print(f"当前时间：{NowTime}/{IteratedTime}")

        # 测量 GetUpTime 的运行时间
        t0 = time.perf_counter()
        Uptime = GetUpTime()  # 得到更新时间
        t1 = time.perf_counter()
        times_GetUpTime.append(t1 - t0)

        NowTime += Uptime  # 添加更新时间

        # 测量 GetUpNode 的运行时间
        t2 = time.perf_counter()
        UpNode = GetUpNode()  # 得到可以更新的节点
        t3 = time.perf_counter()
        times_GetUpNode.append(t3 - t2)

        # 测量 Update 的运行时间
        t4 = time.perf_counter()
        Update(G, NowTime, UpNode)  # 调整当前全局策略
        t5 = time.perf_counter()
        times_Update.append(t5 - t4)

        # 测量 ImitateUpdate 的运行时间
        t6 = time.perf_counter()
        ImitateUpdate(G, UpNode, NowTime)  # 模仿
        t7 = time.perf_counter()
        times_ImitateUpdate.append(t7 - t6)

        # 以下为统计策略数量的已有逻辑
        current_time = math.floor(NowTime)
        if current_time > last_recorded_time:
            last_recorded_time = current_time
            NumOfC = 0
            NumOfD = 0
            NumOfL = 0
            for i in range(IndividualNum):
                if NowStrategiesList[i] == -1:
                    NumOfL += 1
                elif NowStrategiesList[i].theStrategy == 1:
                    NumOfC += 1
                elif NowStrategiesList[i].theStrategy == 0:
                    NumOfD += 1
            times.append(NowTime)
            result_C.append(NumOfC / IndividualNum)
            result_D.append(NumOfD / IndividualNum)
            result_L.append(NumOfL / IndividualNum)

        iteration_count += 1
        if iteration_count >= 50000:
            # 计算各个步骤的平均运行时间
            avg_GetUpTime = sum(times_GetUpTime) / len(times_GetUpTime)
            avg_GetUpNode = sum(times_GetUpNode) / len(times_GetUpNode)
            avg_Update = sum(times_Update) / len(times_Update)
            avg_ImitateUpdate = sum(times_ImitateUpdate) / len(times_ImitateUpdate)

            print("平均 GetUpTime 耗时: {:.6f}秒".format(avg_GetUpTime))
            print("平均 GetUpNode 耗时: {:.6f}秒".format(avg_GetUpNode))
            print("平均 Update 耗时: {:.6f}秒".format(avg_Update))
            print("平均 ImitateUpdate 耗时: {:.6f}秒".format(avg_ImitateUpdate))
            print(f"平均 1次蒙特卡洛 耗时: {IndividualNum*(avg_GetUpTime+avg_GetUpNode+avg_Update+avg_ImitateUpdate):.6f}秒")
            quit()

print((float(sum(result_C[-1000:])/len(result_C[-1000:]))))
print((float(sum(result_D[-1000:])/len(result_C[-1000:]))))
print((float(sum(result_L[-1000:])/len(result_C[-1000:]))))



# 将数据保存到 CSV 文件
filename = "IdeaTestTest.csv"
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # 写入表头
    writer.writerow(['Time', 'C', 'D', 'L'])
    # 写入数据
    for t, c, d, l in zip(times, result_C, result_D, result_L):
        writer.writerow([t, c, d, l])

print(f"数据已保存到 {filename}")

