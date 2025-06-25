import networkx as nx
# import numpy as np
import matplotlib.pyplot as plt
import random

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

num_nodes = 4900
m = 2

num_iterations = 5000 * num_nodes
b = 1.1
R = 1
S = 0.0
T = b
P = 0.0

K = 0.1

G =PeriodicLattice(70,70)
strategies = [random.choice([0, 1]) for _ in range(num_nodes)]


def get_payoff(node):
    payoff = 0
    for neighbor in G.neighbors(node):
        if strategies[node] == 1 and strategies[neighbor] == 1:
            payoff += R
        elif strategies[node] == 1 and strategies[neighbor] == 0:
            payoff += S
        elif strategies[node] == 0 and strategies[neighbor] == 1:
            payoff += T
        else:
            payoff += P
    return payoff


cooperation_levels = []

for i in range(num_iterations):

    new_strategies = strategies.copy()

    node = random.choice(range(0,num_nodes-1))
    neighbors = list(G.neighbors(node))
    if neighbors:
        neighbor = random.choice(neighbors)
        Kd = 4
        prob = ((get_payoff(neighbor) - get_payoff(node))/(b*Kd))
        if random.random() < prob:
            new_strategies[node] = strategies[neighbor]
        strategies = new_strategies

    if i% 10000 == 0:
        print(f"{100*i/num_iterations:.2f}%")
        cooperation_level = sum(strategies)/len(strategies)
        cooperation_levels.append(cooperation_level)

print(sum(cooperation_levels[-30000:])/len(cooperation_levels[-30000:]))

plt.plot(cooperation_levels)
plt.xlabel('Iterations')
plt.ylabel('Cooperation Level')
plt.show()
