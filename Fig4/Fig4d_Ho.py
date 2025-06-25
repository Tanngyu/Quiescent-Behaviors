import csv
import math
import random
import networkx as nx
import numpy as np

class PeriodicLattice:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.G = nx.Graph()
        self.G.add_nodes_from(range(rows * cols))
        for node in self.G.nodes():
            x = node % cols
            y = node // cols
            neighbors = []
            if cols > 1:
                left_x = (x - 1) % cols
                right_x = (x + 1) % cols
                neighbors.append((left_x, y))
                neighbors.append((right_x, y))
            if rows > 1:
                up_y = (y - 1) % rows
                down_y = (y + 1) % rows
                neighbors.append((x, up_y))
                neighbors.append((x, down_y))
            for nx_coord, ny_coord in neighbors:
                neighbor = ny_coord * cols + nx_coord
                self.G.add_edge(node, neighbor)

    def get_average_degree(self):
        degrees = dict(self.G.degree()).values()
        return sum(degrees) / len(degrees) if degrees else 0.0

    def get_all_nodes(self):
        return list(self.G.nodes())

    def get_neighbors(self, node):
        return list(self.G.neighbors(node))

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
            print("Queue is empty")
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

IndividualNum = 2500
IteratedTime = 5000
AverageNum = 1
AverageDegree = 4

G = PeriodicLattice(50,50)

def get_payoff(G, node):
    payoff = 0
    for neighbor in list(G.get_neighbors(node)):
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

def Update(G, NowTime):
    for UpdateNode in G.get_all_nodes():
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
    SelectNeighbors = [neig for neig in G.get_neighbors(UpdateNode) if NowStrategiesList[neig] != -1]
    if len(SelectNeighbors) == 0:
        if NowStrategiesList[UpdateNode] == -1:
            StrategiesQueueList[UpdateNode].enqueue(
                Strategy((StrategiesQueueList[UpdateNode].lastStrategies).theStrategy, random.expovariate(MuList[UpdateNode]), EnqueueTime=NowTime))
            return
        else:
            StrategiesQueueList[UpdateNode].enqueue(
                Strategy(NowStrategiesList[UpdateNode].theStrategy, random.expovariate(MuList[UpdateNode]), EnqueueTime=NowTime))
            return
    Neighbor = random.choice(SelectNeighbors)
    NeighborPayoff = get_payoff(G, Neighbor)
    SelfPayoff = get_payoff(G, UpdateNode)
    Kd = 4
    if random.random() <= ((NeighborPayoff - SelfPayoff)/(b*Kd)):
        StrategiesQueueList[UpdateNode].enqueue(
            Strategy(NowStrategiesList[Neighbor].theStrategy, random.expovariate(MuList[UpdateNode]), EnqueueTime=NowTime))
        return
    else:
        if NowStrategiesList[UpdateNode] == -1:
            StrategiesQueueList[UpdateNode].enqueue(
                Strategy((StrategiesQueueList[UpdateNode].lastStrategies).theStrategy, random.expovariate(MuList[UpdateNode]), EnqueueTime=NowTime))
            return
        else:
            StrategiesQueueList[UpdateNode].enqueue(
                Strategy(NowStrategiesList[UpdateNode].theStrategy, random.expovariate(MuList[UpdateNode]), EnqueueTime=NowTime))
            return

def StrategiesQueueListInit(StrategiesQueueList, NowTime):
    total_queues = len(StrategiesQueueList)
    half = total_queues // 2
    selected_indices = random.sample(range(total_queues), half)
    for index in selected_indices:
        StrategiesQueueList[index].enqueue(Strategy(0, random.expovariate(MuList[index]), EnqueueTime=NowTime))
    for index, queue in enumerate(StrategiesQueueList):
        if index not in selected_indices:
            queue.enqueue(Strategy(1, random.expovariate(MuList[index]), EnqueueTime=NowTime))
    return StrategiesQueueList

def GetUpTime():
    total_lambda = sum(LambdaList)
    uptime = random.expovariate(total_lambda)
    return uptime

def GetUpNode():
    selected_index = random.choices(list(G.get_all_nodes()), weights=LambdaList, k=1)[0]
    return selected_index

b = 1.1
R, S, T, P = 1, 0, b, 0

WaitingTimes = [round(x, 2) for x in np.linspace(0.1, 9, 20)]
sigma_s = [0.1]
heatmap_C = []
heatmap_D = []
heatmap_L = []

for WaitTime in WaitingTimes:
    p = float(WaitTime/(1+WaitTime))
    plot_C = []
    plot_D = []
    plot_L = []
    for sigma in sigma_s:
        result_C = []
        result_D = []
        result_L = []
        times = []
        LambdaList = [p for _ in range(IndividualNum)]
        MuList = [1 for _ in range(IndividualNum)]
        NowStrategiesList = [-1 for _ in range(IndividualNum)]
        NowTime = 0
        StrategiesQueueList = [StrategiesQueue(300) for _ in range(IndividualNum)]
        StrategiesQueueList = StrategiesQueueListInit(StrategiesQueueList, NowTime)
        Update(G, NowTime)
        while NowTime < IteratedTime:
            Uptime = GetUpTime()
            NowTime += Uptime
            Update(G, NowTime)
            UpNode = GetUpNode()
            ImitateUpdate(G, UpNode, NowTime)
            print(f"Current time: {p:.2f}/{WaitTime}/{NowTime}/{IteratedTime}")
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
        plot_C.append(float(sum(result_C[-300*IndividualNum:])/len(result_C[-300*IndividualNum:])))
        plot_D.append(float(sum(result_D[-300*IndividualNum:])/len(result_D[-300*IndividualNum:])))
        plot_L.append(float(sum(result_L[-300*IndividualNum:])/len(result_L[-300*IndividualNum:])))
    heatmap_C.append(plot_C)
    heatmap_D.append(plot_D)
    heatmap_L.append(plot_L)

transposed_C = list(zip(*heatmap_C))
transposed_D = list(zip(*heatmap_D))
transposed_L = list(zip(*heatmap_L))

with open('WaitTimeSL.py.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Type'] + WaitingTimes)
    for row in transposed_C:
        writer.writerow(['C'] + list(row))
    for row in transposed_D:
        writer.writerow(['D'] + list(row))
    for row in transposed_L:
        writer.writerow(['L'] + list(row))