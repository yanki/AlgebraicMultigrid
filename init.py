import sys
import networkx as nx
import matplotlib.pyplot as plt
sys.path.insert(0, 'src')
import alg_methods as method


def PracticeGraph():
    Graph = nx.Graph()
    Graph.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], volume=1)
    Graph.add_edges_from([
        (1, 3), (2, 3), (1, 12), (2, 12),
        (3, 8), (3, 4), (4, 8), (4, 7),
        (4, 6), (4, 5), (5, 6), (6, 9),
        (6, 7), (6, 8), (7, 4), (7, 8),
        (9, 10), (9, 11), (11, 10)])
    return Graph

G = PracticeGraph()
# print nx.Laplacian(G)
print G.nodes(data=True)
print type(G)
print nx.draw(G)
print plt.draw()
# G = method.AlgebraicMultigrid(G)
