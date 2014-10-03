import sys
import time
import networkx as nx
import matplotlib.pyplot as plt
sys.path.insert(0, 'src')
import alg_methods as method

start_time = time.time()

def SimpleGraph(which):
    Graph = nx.Graph()
    # Graph.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], volume=1.0, future_volume=0.0)
    # Graph.add_edges_from([
    #     (1, 3), (2, 3), (1, 12), (2, 12),
    #     (3, 8), (3, 4), (4, 8), (4, 7),
    #     (4, 6), (4, 5), (5, 6), (6, 9),
    #     (6, 7), (6, 8), (7, 8),
    #     (9, 10), (9, 11), (11, 10)], weight=102.0)
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
        Graph.add_nodes_from(range(1, 32), volume=1.0)
        Graph.add_edges_from([
            (1, 7, {'weight': 1.0}), (1, 8, {'weight': 1.0}), (1, 9, {'weight': 1.0}), (1, 10, {'weight': 1.0}),
            (2, 11, {'weight': 1.0}), (2, 12, {'weight': 1.0}), (2, 13, {'weight': 1.0}), (2, 14, {'weight': 1.0}),
            (3, 15, {'weight': 1.0}), (3, 16, {'weight': 1.0}), (3, 17, {'weight': 1.0}), (3, 18, {'weight': 1.0}),
            (4, 19, {'weight': 1.0}), (4, 20, {'weight': 1.0}), (4, 21, {'weight': 1.0}), (4, 22, {'weight': 1.0}),
            (5, 23, {'weight': 1.0}), (5, 24, {'weight': 1.0}), (5, 25, {'weight': 1.0}), (5, 26, {'weight': 1.0}),
            (6, 27, {'weight': 1.0}), (6, 28, {'weight': 1.0}), (6, 29, {'weight': 1.0}), (6, 30, {'weight': 1.0}),
            (1, 2, {'weight': 1.0}), (2, 3, {'weight': 1.0}), (3, 4, {'weight': 1.0}),
            (4, 5, {'weight': 1.0}), (5, 6, {'weight': 1.0}), (6, 1, {'weight': 1.0}),
            (31, 1, {'weight': 1.0}), (31, 2, {'weight': 1.0}), (31, 3, {'weight': 1.0}),
            (31, 4, {'weight': 1.0}), (31, 5, {'weight': 1.0}), (31, 6, {'weight': 1.0})])
    else:
        sys.exit("ERROR: No Graph Selected.")
    return Graph


def DrawGraph(Graph):
    layout = nx.spring_layout(G)
    nx.draw(G, layout, node_color='#A0CBE2', edge_color='#BB0000', width=2, edge_cmap=plt.cm.Blues, with_labels=True)
    plt.savefig("graph.png", dpi=1000, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1)

G = SimpleGraph(3)
DrawGraph(G)
# print G.nodes(data=True)
# for node, volume in (nx.get_node_attributes(G, 'volume')).iteritems():
#     print "Node: " + str(node) + " Volume: " + str(volume)
G = method.AlgebraicMultigrid(G)
print "Execution time: " + str(time.time() - start_time) + " seconds"
