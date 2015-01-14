import networkx as nx
import scipy.sparse as sps
import settings
import operator
import numpy as np
import math

Q = settings.threshhold
r = settings.neighborhood_size
n = settings.aggregate_size
a = settings.edge_size
laplacian = None
volume = None

__all__ = ['AlgebraicMultigrid']

def AlgebraicMultigrid(G, **args):
    """Accept a graph and attempt to solve it."""

    global volume

    volume = args['volume']
    if volume is None:
        G.nodes()  # set all node volumes to 1

    if nx.number_of_nodes(G) <= 7:
        G = Solve(G)
    else:
        # seeds = dict(seeds.items() + out.items())
        seeds = GetCoarseSeeds(G)
        # GetCoarseMatrix(G, seeds, len(G.nodes()))
        # print "SEEDS: " + str(seeds)
        print "num of seeds: " + str(len(seeds)) + " num of nodes: " + str(len(G.nodes()))
        """
        1. find algebraic distance
        2. future volume
        3. coarse nodes
        4. coarse edges
        """
        # S = AlgebraicMultigrid(G)
        # S = Refine(S)
    return G

def GetCoarseMatrix(G, seeds, nodes_num):
    """Gets the matrix of a coarse graph."""
    global a
    global laplacian

    row_array = []
    col_array = []
    data_array = []
    for index, i in enumerate(seeds):
        for j in range(nodes_num):
            if i == (j + 1):
                row_array.append(index)
                col_array.append(j)
                data_array.append(1)
            elif j + 1 in G.neighbors(i):
                row_array.append(index)
                col_array.append(j)
                edge_sum = 0.0
                for edge in (x for x in G.neighbors(j + 1) if x in seeds):
                    if G.edge[i][j + 1]['weight'] > a:
                        edge_sum += 1.0
                # data_array.append(math.ceil((G.edge[i][j + 1]['weight'] / edge_sum) * 100) / 100)
                data_array.append(G.edge[i][j + 1]['weight'] / edge_sum)
            else:
                continue
    row = np.array(row_array)
    col = np.array(col_array)
    data = np.array(data_array)
    mtx = sps.csr_matrix((data, (col, row)),
        shape=(nodes_num, len(seeds)))
    # print mtx.todense()
    # print (((mtx.transpose()).dot(laplacian)).dot(mtx)).todense()

def FutureVolume(G, nodes):
    """Coarsens the graph G."""

    global Q
    global r
    global laplacian
    global volume

    if(nodes is None):
        node_list = G.nodes()
    else:
        node_list = nodes
    for i in node_list:
        G.node[i]['future_volume'] = G.node[i][volume]
        for j in (x for x in G.neighbors(i) if x in node_list):
            degree = G.degree(j) * 1.0
            adjacency = degree / min(r, Q * degree)
            sum_weight = 0.0
            for k in G.neighbors(j):
                sum_weight += G.edge[j][k]['weight']
            norm_weight = G.edge[i][j]['weight'] / sum_weight
            G.node[i]['future_volume'] += G.node[j][volume] * min(1.0, adjacency * norm_weight)
    data_list = []
    for node in G.nodes():
        data_list.append(G.node[node]['future_volume'])
        print "Node: " + str(node) + " Neighbors: " + str(G.neighbors(node)) + " Future Volume: " + str(G.node[node]['future_volume'])
    return G


def GetCoarseSeeds(G):
    """Returns a list of nodes that are chosen as seeds for coarse graph."""

    global n
    global Q
    global laplacian

    laplacian = nx.laplacian_matrix(G)
    G = FutureVolume(G, None)
    total_volume = 0.0
    seeds = {}
    non_seeds = {}
    for i in nx.nodes_iter(G):
        total_volume += G.node[i]['future_volume']
        non_seeds[i] = G.node[i]['future_volume']
    avg_volume = total_volume / nx.number_of_nodes(G)
    # print avg_volume
    for node, value in non_seeds.items():
        if(value > n * avg_volume):
            seeds[node] = value
            del non_seeds[node]
    # print seeds
    # exit()
    G = FutureVolume(G, non_seeds.keys())
    for node in non_seeds.keys():
        non_seeds[node] = G.node[node]['future_volume']
    sorted_nonseeds = sorted(non_seeds.items(), key=operator.itemgetter(1), reverse=True)
    print sorted_nonseeds
    for node, value in sorted_nonseeds:
        seed_edges = 0.0
        total_edges = 0.0
        # print "Node: " + str(node) + " has neighbors: " + str(G.neighbors(node))
        for neighbor in G.neighbors(node):
            total_edges += G.edge[node][neighbor]['weight']
            if neighbor in seeds:
                # print "WHATAOIDNFOADNOFISDHNF"
                seed_edges += G.edge[node][neighbor]['weight']
        if seed_edges / total_edges <= Q:
            seeds[node] = value
            del non_seeds[node]
    # print seeds
    return seeds.keys()

def DefineVariables():
    """Defines the variables for the Algorithm."""

def Solve(G):
    """Solve the graph."""
    print "In Solve."
    return None


def Refine(S):
    """Work back up the V-graph."""
    print "In refine."
    return None
