import random
import networkx as nx
import matplotlib.pyplot as plt

graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'C': 2, 'D': 5},
    'C': {'D': 1},
    'D': {}
}

def hill_climb_path(graph, start, goal):
    current_node = start
    path = [current_node]

    while current_node != goal:
        neighbors = graph[current_node]
        if not neighbors:
            return path, float('inf')

        next_node = min(neighbors, key=neighbors.get)
        path.append(next_node)
        current_node = next_node

    total_cost = sum(graph[path[i]][path[i + 1]] for i in range(len(path) - 1))
    return path, total_cost

def draw_graph(graph, path=None):
    G = nx.DiGraph()

    for node, neighbors in graph.items():
        for neighbor, cost in neighbors.items():
            G.add_edge(node, neighbor, weight=cost)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=15, font_weight="bold", arrows=True)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=12)

    if path:
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=3)

    plt.show()

start_node = 'A'
goal_node = 'D'
path, cost = hill_climb_path(graph, start_node, goal_node)
print(f"Final path: {path}, Total cost: {cost}")
draw_graph(graph, path)
