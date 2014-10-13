import sys
import time
import networkx as nx
import matplotlib.pyplot as plt
sys.path.insert(0, 'src')
from alg_methods import AlgebraicMultigrid as AM
from networkx import NetworkXError

__author__ = """\n""".join(['Ilya Safro <isafro@g.clemson.edu>',
                            'Mikita Yankouski <myankou@g.clemson.edu>',
                            'Tiffany Verkaik <tverkai@g.clemson.edu>'])

start_time = time.time()

def SimpleGraph(which):
    Graph = nx.Graph()
    if which == 1:
        Graph.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8], volume=1.0)
        Graph.add_edges_from([
            (1, 3, {'weight': 10.0}), (2, 3, {'weight': 10.0}),
            (3, 4, {'weight': 30.0}), (4, 8, {'weight': 20.0}),
            (4, 7, {'weight': 10.0}), (4, 6, {'weight': 30.0}),
            (4, 5, {'weight': 20.0}), (5, 6, {'weight': 20.0}),
            (6, 7, {'weight': 20.0}), (6, 8, {'weight': 10.0}),
            (7, 8, {'weight': 20.0})])
    elif which == 2:
        Graph.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8], volume=1.0)
        Graph.add_edges_from([
            (1, 3), (2, 3),
            (3, 4), (4, 8),
            (4, 7), (4, 6),
            (4, 5), (5, 6),
            (6, 7), (6, 8),
            (7, 8)], weight=1.0)
    elif which == 3:
        Graph.add_nodes_from(range(1, 33), volume=1.0)
        Graph.add_edges_from([
            (1, 5, {'weight': 1.0}), (1, 6, {'weight': 1.0}), (1, 7, {'weight': 1.0}), (1, 8, {'weight': 1.0}),
            (1, 2, {'weight': 1.0}), (1, 4, {'weight': 1.0}), (1, 3, {'weight': 1.0}),
            (2, 23, {'weight': 1.0}), (2, 24, {'weight': 1.0}), (2, 25, {'weight': 1.0}), (2, 26, {'weight': 1.0}), (2, 27, {'weight': 1.0}),
            (2, 28, {'weight': 1.0}), (2, 29, {'weight': 1.0}), (2, 30, {'weight': 1.0}), (2, 31, {'weight': 1.0}), (2, 32, {'weight': 1.0}),
            (2, 3, {'weight': 1.0}),
            (3, 9, {'weight': 1.0}), (3, 10, {'weight': 1.0}), (3, 11, {'weight': 1.0}), (3, 12, {'weight': 1.0}),
            (3, 4, {'weight': 1.0}),
            (4, 13, {'weight': 1.0}), (4, 14, {'weight': 1.0}), (4, 15, {'weight': 1.0}), (4, 16, {'weight': 1.0}), (4, 17, {'weight': 1.0}),
            (4, 18, {'weight': 1.0}), (4, 19, {'weight': 1.0}), (4, 20, {'weight': 1.0}), (4, 21, {'weight': 1.0}), (4, 22, {'weight': 1.0})])
    else:
        sys.exit("ERROR: No Graph Selected.")
    return Graph


def DrawGraph(Graph):
    layout = nx.spring_layout(Graph, iterations=2000)
    nx.draw(Graph, layout, node_color='#A0CBE2', edge_color='#BB0000', width=2, edge_cmap=plt.cm.Blues, with_labels=True)
    plt.savefig("graph.png", dpi=1000, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1)

G = SimpleGraph(3)
# DrawGraph(G)
if (G.is_directed() is True) or (G.is_multigraph() is True):
        raise NetworkXError("AlgebraicMultigrid is not defined for directed or multi graphs.")
elif (len(G.nodes()) == 0) or (len(G.edges()) == 0):
        raise NetworkXError("The graph does not contain either Nodes or Edges.")
else:
    AM(G)
print "Execution time: " + str(time.time() - start_time) + " seconds"
