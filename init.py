import sys
import time
import networkx as nx
import matplotlib.pyplot as plt
import simplejson as json
import scipy.io as sci
sys.path.insert(0, 'src')
from alg_methods import AlgebraicMultigrid as AM
from networkx import NetworkXError
from networkx.readwrite import json_graph

__author__ = """\n""".join(['Ilya Safro <isafro@g.clemson.edu>',
                            'Mikita Yankouski <myankou@g.clemson.edu>',
                            'Tiffany Verkaik <tverkai@g.clemson.edu>'])

start_time = time.time()

def readMesh():
    edge_list = []
    G = nx.Graph()
    with open('graphs/mesh33x33.rmf') as f:
        for index, line in enumerate(f.readlines()):
            values = line.split(' ')
            edge = (int(values[1]), int(values[2]))
            edge_list.append(edge)
    G.add_edges_from(edge_list, weight=1)
    return G

def DrawGraph(Graph):
    layout = nx.random_layout(Graph, dim=2)
    # layout = nx.spring_layout(Graph, iterations=2000)
    nx.draw(Graph, layout, node_color='#A0CBE2', edge_color='#BB0000', width=2, edge_cmap=plt.cm.Blues, with_labels=True)
    plt.savefig("graph.png", dpi=1500, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1)

# G = nx.Graph()
# G = json_graph.node_link_graph(json.load(open('graphs/star.json')))
# matrix = sci.mmread('graphs/mesh.mtx')
# G = nx.from_scipy_sparse_matrix(matrix)
G = readMesh()
for node in G.nodes():
    G.node[node]['volume'] = 1
# DrawGraph(G)
if (G.is_directed() is True) or (G.is_multigraph() is True):
        raise NetworkXError("AlgebraicMultigrid is not defined for directed or multi graphs.")
elif (len(G.nodes()) == 0) or (len(G.edges()) == 0):
        raise NetworkXError("The graph does not contain either Nodes or Edges.")
else:
    # data_list = []
    # for node in G.nodes():
    #     data_list.append(len(G.neighbors(node)))
    # print data_list
    # json.dump(json_graph.node_link_data(G), open('graph.json', 'w'))
    AM(G, volume='volume')
print "Execution time: " + str(time.time() - start_time) + " seconds"
