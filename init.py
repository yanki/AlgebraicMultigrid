import sys
import time
import networkx as nx
import matplotlib.pyplot as plt
import simplejson as json
sys.path.insert(0, 'src')
from alg_methods import AlgebraicMultigrid as AM
from networkx import NetworkXError
from networkx.readwrite import json_graph

__author__ = """\n""".join(['Ilya Safro <isafro@g.clemson.edu>',
                            'Mikita Yankouski <myankou@g.clemson.edu>',
                            'Tiffany Verkaik <tverkai@g.clemson.edu>'])

start_time = time.time()

def DrawGraph(Graph):
    layout = nx.spring_layout(Graph, iterations=10000)
    nx.draw(Graph, layout, node_color='#A0CBE2', edge_color='#BB0000', width=2, edge_cmap=plt.cm.Blues, with_labels=True)
    plt.savefig("graph.png", dpi=1000, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1)

G = nx.Graph()
data = json.load(open('graph.json'))
G = json_graph.node_link_graph(data)
DrawGraph(G)
if (G.is_directed() is True) or (G.is_multigraph() is True):
        raise NetworkXError("AlgebraicMultigrid is not defined for directed or multi graphs.")
elif (len(G.nodes()) == 0) or (len(G.edges()) == 0):
        raise NetworkXError("The graph does not contain either Nodes or Edges.")
else:
    # json.dump(json_graph.node_link_data(G), open('graph.json', 'w'))
    AM(G)
print "Execution time: " + str(time.time() - start_time) + " seconds"
