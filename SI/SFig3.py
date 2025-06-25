import bisect
import csv
import math
import random
import time
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt, rcParams
from scipy.stats import truncnorm, expon
from scipy.special import iv

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

    def get_length(self):
        if self.front == -1:
            return 0
        elif self.rear >= self.front:
            return self.rear - self.front + 1
        else:
            return (self.size - self.front) + (self.rear + 1)

IndividualNum = 5000
AverageNum = 1
AverageDegree = 4

G = nx.watts_strogatz_graph(IndividualNum, AverageDegree, 0.2)
p = 0.5

LambdaList = [p for _ in range(IndividualNum)]
MuList = [float(1) for _ in range(IndividualNum)]
total_lambda = sum(LambdaList)

C_crital = (1/p) - 1

b = 1.2
c = 1
sigma = 0.1
R, S, T, P = 1, 0, b, 0

IteratedStep = 250
IteratedTime = IteratedStep

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
        StrategiesQueueList[index].enqueue(Strategy(1, 1/MuList[index], EnqueueTime=NowTime))

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

def mm1_transient_bessel(t_list, mu=1.0, N=300, j_max=800, eps=1e-10):
    t_list = np.atleast_1d(t_list)
    lam = p
    rho = p

    max_k = N + j_max

    lengths = []
    for t in t_list:
        x = 2 * t * np.sqrt(lam * mu)
        common = np.exp(-(lam + mu) * t)

        I = iv(np.arange(max_k+1), x)

        queue_len_sum = 0.0
        prob_sum = 0.0
        for n in range(N+1):
            part1 = rho**(n/2) * I[n] if n >= 0 else 0.0
            part2 = rho**((n+1)/2) * I[n+1]
            s = 0.0
            const = (1 - rho) * rho**n
            for j in range(n+2, max_k+1):
                term = const * rho**(-j/2) * I[j]
                s += term
                if abs(term) < eps:
                    break

            p_n = common * (part1 + part2 + s)
            prob_sum += p_n
            queue_len = max(n-1, 0)
            queue_len_sum += queue_len * p_n

        queue_len_sum /= prob_sum
        lengths.append(queue_len_sum)

    return lengths

for i in range(AverageNum):
    NUM_Strategies = []
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
        Update(G, NowTime,UpNode)
        ImitateUpdate(G, UpNode, NowTime)

        current_time = math.floor(NowTime)
        if current_time > last_recorded_time:
            UpdateALL(G, NowTime)
            if current_time % 1 == 0:
                print(f"Current time: {NowTime}/{IteratedTime}")
            last_recorded_time = current_time
            NumOfQL = 0
            for i in range(IndividualNum):
                NumOfQL += StrategiesQueueList[i].get_length()
            times.append(NowTime)
            NUM_Strategies.append(NumOfQL / IndividualNum)

THE_QL = mm1_transient_bessel(times)

print(THE_QL)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 22
rcParams['mathtext.fontset'] = 'cm'
rcParams['mathtext.rm'] = 'serif'
rcParams['mathtext.it'] = 'serif:italic'
rcParams['mathtext.bf'] = 'serif:bold'
print(len(NUM_Strategies))
print(np.mean(NUM_Strategies[-100:]))

plt.figure(figsize=(10, 6))
plt.scatter(times, NUM_Strategies, label=r'Numerical $p_n(t)$', color='#2D85F0', linewidth=2, facecolors='none')
plt.plot(times, THE_QL, label=r'Theroetical $p_n(t)$', color='#F4433C', linestyle="--", linewidth=5)

plt.axhline(np.mean(NUM_Strategies[-50:]), color="#2D85F0", linestyle="--", linewidth=3, alpha=0.7)
plt.axhline(np.mean(THE_QL[-50:]), color="#F4433C", linestyle="--", linewidth=3, alpha=0.7)
plt.axhline((p**2)/(1-p), color="#238B45", linestyle="--",label=r'Theroetical $p_n$', linewidth=3, alpha=0.7)

plt.xlabel(r'$t$')
plt.ylabel(r'$p_n(t)$')
plt.legend(fontsize=22, markerscale=2)
plt.tight_layout()
plt.show()