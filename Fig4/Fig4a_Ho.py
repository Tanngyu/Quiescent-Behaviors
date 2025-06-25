import bisect
import csv
import math
import random
import networkx as nx

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

    def nodes(self):
        return list(self.G.nodes())

    def neighbors(self, node):
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

IndividualNum = 4900
AverageNum = 1
AverageDegree = 4

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
    for UpdateNode in G.nodes():
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
    Kd = 4
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

Rhos = [round(i * 0.05, 2) for i in range(1, 20)]
sigma_s = [0.1]
heatmap_C = []
heatmap_D = []
heatmap_L = []

for p in Rhos:
    plot_C = []
    plot_D = []
    plot_L = []
    for sigma in sigma_s:
        G = PeriodicLattice(70,70)
        result_C = []
        times = []
        LambdaList = [p for _ in range(IndividualNum)]
        MuList = [1 for _ in range(IndividualNum)]
        nodes_list = list(G.nodes())
        cumulative_weights = []
        total = 0
        for w in LambdaList:
            total += w
            cumulative_weights.append(total)
        total_lambda = sum(LambdaList)
        IteratedStep = 10000
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
                if current_time % 100 == 0:
                    print(f"Current time: {p}/{NowTime}/{IteratedTime}")
                last_recorded_time = current_time
                NumOfC = 0
                NumOfD = 0
                for i in [I for I in range(IndividualNum) if NowStrategiesList[I] != -1]:
                    if NowStrategiesList[i].theStrategy == 1:
                        NumOfC += 1
                    if NowStrategiesList[i].theStrategy == 0:
                        NumOfD += 1
                times.append(NowTime)
                result_C.append(NumOfC / (NumOfC+NumOfD))
        plot_C.append(sum(result_C[-1000:]) / len(result_C[-1000:]))
    heatmap_C.append(plot_C)

with open('Theory_1_Ho.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['rho/sigma'] + Rhos)
    transposed = list(zip(*heatmap_C))
    for i, sigma in enumerate(sigma_s):
        row = [sigma] + list(transposed[i])
        writer.writerow(row)
