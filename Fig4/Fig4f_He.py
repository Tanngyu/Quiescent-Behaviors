import bisect
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

G = nx.barabasi_albert_graph(IndividualNum,2)

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
    SelectNeighbors = [neig for neig in nx.neighbors(G, UpdateNode) if NowStrategiesList[neig] != -1]

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

    if random.random() <= ((NeighborPayoff - SelfPayoff)/(b*Kd)):
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
        StrategiesQueueList[index].enqueue(Strategy(0, random.expovariate(MuList[index]) , EnqueueTime=NowTime))

    for index, queue in enumerate(StrategiesQueueList):
        if index not in selected_indices:
            queue.enqueue(Strategy(1, random.expovariate(MuList[index]) , EnqueueTime=NowTime))

    return StrategiesQueueList

def GetUpTime():
    total_lambda = sum(LambdaList)
    uptime = random.expovariate(total_lambda)
    return uptime

def GetUpNode():
    r = random.uniform(0, total)
    idx = bisect.bisect(cumulative_weights, r)
    return nodes_list[idx]

b = 1.8
R, S, T, P = 1, 0, b, 0

sigma_s = [0.1]
WaitingTimes = [0.1, 0.57, 1.04, 1.51, 1.97, 2.44, 2.91, 3.38, 3.85, 4.32, 4.78, 5.25, 5.72, 6.19, 6.66, 7.13, 7.59, 8.06, 8.53, 9]
R, S, T, P = 1, 0, b, 0
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

        nodes_list = list(G.nodes)
        cumulative_weights = []
        total = 0
        for w in LambdaList:
            total += w
            cumulative_weights.append(total)

        total_lambda = sum(LambdaList)
        IteratedStep = 5000
        IteratedTime = ((1 / total_lambda) * IndividualNum * IteratedStep)

        NowStrategiesList = [-1 for _ in range(IndividualNum)]

        NowTime = 0

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
                print(f"Current time: {p:.2f}/{WaitTime}/{NowTime}/{IteratedTime}")
                last_recorded_time = current_time

                for i in range(IndividualNum):
                    if NowStrategiesList[i] == -1:
                        continue
                    else:
                        result_C.append(NowStrategiesList[i].UsedTime - NowStrategiesList[i].EnqueueTime)
                times.append(NowTime)

        plot_C.append(sum(result_C[-3000:]) / len(result_C[-3000:]))

        heatmap_C.append(plot_C)

    transposed_C = list(zip(*heatmap_C))

    with open('WT_theory.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(['Type'] + WaitingTimes)

        for row in transposed_C:
            writer.writerow(['C'] + list(row))