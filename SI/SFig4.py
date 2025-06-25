import bisect
import csv
import math
import random
import time
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt, rcParams
from scipy.stats import truncnorm, expon

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
            print("Empty queue")
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

IndividualNum = 5000
AverageNum = 1
AverageDegree = 4

G = nx.watts_strogatz_graph(IndividualNum, AverageDegree, 0.2)
p = 0.5

LambdaList = [float(0.5) for _ in range(IndividualNum)]
MuList = [float(1) for _ in range(IndividualNum)]
total_lambda = sum(LambdaList)

C_crital = (1/p) - 1

b = 1.2
c = 1
sigma = 0.1
R, S, T, P = 1, 0, b, 0

IteratedStep = 5000
IteratedTime = ((1/total_lambda) * IndividualNum * IteratedStep)

nodes_list = list(G.nodes)
cumulative_weights = []
total = 0
for w in LambdaList:
    total += w
    cumulative_weights.append(total)

def get_payoff(G, node):
    payoff = 0
    for neighbor in list(G.neighbors(node)):
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

def UpdateALL(G, NowTime):
    for UpdateNode in G.nodes:
        if NowStrategiesList[UpdateNode] == -1:
            if StrategiesQueueList[UpdateNode].is_empty():
                continue

            NowStrategiesList[UpdateNode] = StrategiesQueueList[UpdateNode].dequeue((StrategiesQueueList[UpdateNode].peekFront()).EnqueueTime)

            while (NowTime - NowStrategiesList[UpdateNode].StateTime >= NowStrategiesList[UpdateNode].UsedTime):
                if StrategiesQueueList[UpdateNode].is_empty():
                    NowStrategiesList[UpdateNode] = -1
                    break

                if (StrategiesQueueList[UpdateNode].lastStrategies).UsedTime +(StrategiesQueueList[UpdateNode].lastStrategies).StateTime < (StrategiesQueueList[UpdateNode].peekFront()).EnqueueTime:
                    NowStrategiesList[UpdateNode] = StrategiesQueueList[UpdateNode].dequeue((StrategiesQueueList[UpdateNode].peekFront()).EnqueueTime)
                else:
                    NowStrategiesList[UpdateNode] = StrategiesQueueList[UpdateNode].dequeue((StrategiesQueueList[UpdateNode].lastStrategies).UsedTime + (StrategiesQueueList[UpdateNode].lastStrategies).StateTime)

                if NowStrategiesList[UpdateNode] == -1:
                    break
        else:
            while (NowTime - NowStrategiesList[UpdateNode].StateTime >= NowStrategiesList[UpdateNode].UsedTime):
                NowStrategiesList[UpdateNode] = StrategiesQueueList[UpdateNode].dequeue((StrategiesQueueList[UpdateNode].lastStrategies).UsedTime +(StrategiesQueueList[UpdateNode].lastStrategies).StateTime)
                if NowStrategiesList[UpdateNode] == -1:
                    break

def Update(G, NowTime, UpNode):
    UP_nodes = set(G.neighbors(UpNode))
    for node in UP_nodes.copy():
        UP_nodes.update(G.neighbors(node))
    UP_nodes = list(UP_nodes)

    for UpdateNode in UP_nodes:
        if NowStrategiesList[UpdateNode] == -1:
            if StrategiesQueueList[UpdateNode].is_empty():
                continue

            NowStrategiesList[UpdateNode] = StrategiesQueueList[UpdateNode].dequeue((StrategiesQueueList[UpdateNode].peekFront()).EnqueueTime)

            while (NowTime - NowStrategiesList[UpdateNode].StateTime >= NowStrategiesList[UpdateNode].UsedTime):
                if StrategiesQueueList[UpdateNode].is_empty():
                    NowStrategiesList[UpdateNode] = -1
                    break

                if (StrategiesQueueList[UpdateNode].lastStrategies).UsedTime +(StrategiesQueueList[UpdateNode].lastStrategies).StateTime < (StrategiesQueueList[UpdateNode].peekFront()).EnqueueTime:
                    NowStrategiesList[UpdateNode] = StrategiesQueueList[UpdateNode].dequeue((StrategiesQueueList[UpdateNode].peekFront()).EnqueueTime)
                else:
                    NowStrategiesList[UpdateNode] = StrategiesQueueList[UpdateNode].dequeue((StrategiesQueueList[UpdateNode].lastStrategies).UsedTime + (StrategiesQueueList[UpdateNode].lastStrategies).StateTime)

                if NowStrategiesList[UpdateNode] == -1:
                    break
        else:
            while (NowTime - NowStrategiesList[UpdateNode].StateTime >= NowStrategiesList[UpdateNode].UsedTime):
                NowStrategiesList[UpdateNode] = StrategiesQueueList[UpdateNode].dequeue((StrategiesQueueList[UpdateNode].lastStrategies).UsedTime +(StrategiesQueueList[UpdateNode].lastStrategies).StateTime)
                if NowStrategiesList[UpdateNode] == -1:
                    break

def ImitateUpdate(G, UpdateNode, NowTime):
    SelectNeighbors = [n for n in G[UpdateNode] if NowStrategiesList[n] != -1]

    if len(SelectNeighbors) == 0:
        if NowStrategiesList[UpdateNode] == -1:
            StrategiesQueueList[UpdateNode].enqueue(
                Strategy((StrategiesQueueList[UpdateNode].lastStrategies).theStrategy, random.expovariate(MuList[UpdateNode]) , EnqueueTime=NowTime))
            return
        else:
            StrategiesQueueList[UpdateNode].enqueue(
                Strategy(NowStrategiesList[UpdateNode].theStrategy, random.expovariate(MuList[UpdateNode]) , EnqueueTime=NowTime))
            return

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
    else:
        if NowStrategiesList[UpdateNode] == -1:
            StrategiesQueueList[UpdateNode].enqueue(
                Strategy((StrategiesQueueList[UpdateNode].lastStrategies).theStrategy, random.expovariate(MuList[UpdateNode]) , EnqueueTime=NowTime))
            return
        else:
            StrategiesQueueList[UpdateNode].enqueue(
                Strategy(NowStrategiesList[UpdateNode].theStrategy, random.expovariate(MuList[UpdateNode]) , EnqueueTime=NowTime))
            return

def StrategiesQueueListInit(StrategiesQueueList, NowTime):
    total_queues = len(StrategiesQueueList)
    half = total_queues // 2

    selected_indices = random.sample(range(total_queues), half)
    for index in selected_indices:
        StrategiesQueueList[index].enqueue(Strategy(1, 1/MuList[index] , EnqueueTime=NowTime))

    for index, queue in enumerate(StrategiesQueueList):
        if index not in selected_indices:
            queue.enqueue(Strategy(0, 1/MuList[index] , EnqueueTime=NowTime))

    return StrategiesQueueList

def GetUpTime():
    uptime = random.expovariate(total_lambda)
    return uptime

def GetUpNode():
    r = random.uniform(0, total)
    idx = bisect.bisect(cumulative_weights, r)
    return nodes_list[idx]

def test_C_critical():
    is_satisfy = 0

    sum_p_C = 0
    sum_p_D = 0
    sum_p_Q = 0
    sum_q_C = 0
    sum_q_D = 0
    sum_q_Q = 0

    selected_neighbors = {node: [n for n in G[node] if NowStrategiesList[n] != -1] for node in G.nodes}
    payoffs = {node: get_payoff(G, node) for node in G.nodes}

    for individual in G.nodes:
        SelectNeighbors = selected_neighbors[individual]

        if NowStrategiesList[individual] == -1 or len(SelectNeighbors) == 0:
            if StrategiesQueueList[individual].lastStrategies.theStrategy == 0:
                sum_q_Q += 1
            elif StrategiesQueueList[individual].lastStrategies.theStrategy == 1:
                sum_p_Q += 1
            else:
                print("Error")
                exit()
            continue

        Neighbor = random.choice(SelectNeighbors)
        NeighborPayoff = payoffs[Neighbor]
        SelfPayoff = payoffs[individual]

        Kd = max(G.degree[Neighbor], G.degree[individual])
        probability = (NeighborPayoff - SelfPayoff) / (b * Kd)

        if NowStrategiesList[Neighbor].theStrategy == 1:
            if NowStrategiesList[individual].theStrategy == 0:
                sum_p_D += probability
                sum_q_D += 1 - probability
            if NowStrategiesList[individual].theStrategy == 1:
                sum_p_C += 1
            continue

        if NowStrategiesList[Neighbor].theStrategy == 0:
            if NowStrategiesList[individual].theStrategy == 1:
                sum_q_C += probability
                sum_p_C += 1 - probability
            if NowStrategiesList[individual].theStrategy == 0:
                sum_q_D += 1
            continue

    P_C = sum_p_C / IndividualNum
    P_D = sum_p_D / IndividualNum
    P_Q = sum_p_Q / IndividualNum
    Q_C = sum_q_C / IndividualNum
    Q_D = sum_q_D / IndividualNum
    Q_Q = sum_q_Q / IndividualNum

    if P_Q > Q_Q:
        if (Q_C - P_D) / (P_Q - Q_Q) <= C_crital:
            is_satisfy = 1
    elif P_Q < Q_Q:
        if (Q_C - P_D) / (P_Q - Q_Q) >= C_crital:
            is_satisfy = 1
    else:
        is_satisfy = 0

    return is_satisfy

for i in range(AverageNum):
    result_C = []
    result_D = []
    result_L = []
    times = []
    satisfy = []

    NowStrategiesList = [-1 for _ in range(IndividualNum)]
    NowTime = 1

    StrategiesQueueList = [StrategiesQueue(300) for _ in range(IndividualNum)]
    StrategiesQueueList = StrategiesQueueListInit(StrategiesQueueList, NowTime)

    UpdateALL(G, NowTime)
    last_recorded_time = -1

    while NowTime < IteratedTime:
        Uptime = GetUpTime()
        NowTime += Uptime
        UpNode = GetUpNode()
        Update(G, NowTime, UpNode)
        ImitateUpdate(G, UpNode, NowTime)

        current_time = math.floor(NowTime)
        if current_time > last_recorded_time:
            UpdateALL(G, NowTime)
            if current_time % 100 == 0:
                print(f"Current time: {NowTime}/{IteratedTime}")

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

print((float(sum(result_C[-3000:])/len(result_C[-3000:])))
print((float(sum(result_D[-3000:])/len(result_C[-3000:])))
print((float(sum(result_L[-3000:])/len(result_C[-3000:])))

filename = "IdeaTest1.csv"
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time', 'C', 'D', 'L'])
    for t, c, d, l in zip(times, result_C, result_D, result_L):
        writer.writerow([t, c, d, l])

print(f"Data saved to {filename}")

print(result_C)
print(satisfy)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 22
rcParams['mathtext.fontset'] = 'cm'
rcParams['mathtext.rm'] = 'serif'
rcParams['mathtext.it'] = 'serif:italic'
rcParams['mathtext.bf'] = 'serif:bold'

plt.figure(figsize=(10, 6))
plt.plot(times, result_C, label=r'$C$', color='#2D85F0', linewidth=5)
plt.plot(times, result_D, label=r'$D$', color='#F4433C', linewidth=5)
plt.plot(times, result_L, label=r'$Q$', color='#FFBC32', linewidth=5)

plt.xscale('log')

plt.xlabel(r'$t$')
plt.ylabel(r'$f_S$')
plt.legend(fontsize=22, markerscale=2)
plt.tight_layout()
plt.show()