import sys
import networkx as nx
import matplotlib.pyplot as plt
sys.path.insert(0, 'src')
import alg_methods as method


def SimpleGraph():
    Graph = nx.Graph()
    Graph.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], volume=1.0, future_volume=0.0)
    Graph.add_edges_from([
        (1, 3), (2, 3), (1, 12), (2, 12),
        (3, 8), (3, 4), (4, 8), (4, 7),
        (4, 6), (4, 5), (5, 6), (6, 9),
        (6, 7), (6, 8), (7, 4), (7, 8),
        (9, 10), (9, 11), (11, 10)], weight=102.0)
    return Graph


def DrawGraph(Graph):
    layout = nx.spring_layout(G)
    nx.draw(G, layout, node_color='#A0CBE2', edge_color='#BB0000', width=2, edge_cmap=plt.cm.Blues, with_labels=True)
    plt.savefig("graph.png", dpi=1000, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1)


G = SimpleGraph()
DrawGraph(G)

# print G.nodes(data=True)
G = method.AlgebraicMultigrid(G)
