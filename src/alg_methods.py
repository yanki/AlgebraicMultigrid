import sys
import time
import networkx as nx
import scipy.sparse as sps
# import scipy
import operator
import numpy as np
import matplotlib.pyplot as plt

r = 2.0
Q = .5
n = 2.0
a = 0.0
node_limit = 8
neighborhood_size = 2

laplacian = None
volume = None
iteration = 1
start_time = None

__all__ = ['AlgebraicMultigrid']

def AlgebraicMultigrid(G, **args):
    """Accept a graph and attempt to solve it."""

    global volume
    global laplacian
    global iteration
    global node_limit
    global start_time

    if 'volume' in args:
        volume = args['volume']
    else:
        volume = 'volume'

    print "Iteration: " + str(iteration)
    print "Nodes: " + str(len(G.nodes()))
    print "Edges: " + str(len(G.edges()))
    # nx.draw(G, node_size=50, layout="sfdp")
    # plt.savefig("plots/graph" + str(iteration) + ".png", format="PNG")
    # plt.show()
    # if iteration > 2:
    #     sys.exit()
    if iteration == 1:
        start_time = time.time()
    iteration += 1
    if nx.number_of_nodes(G) <= node_limit:
        G = Solve(G)
    else:
        seeds = GetCoarseSeeds(G)

        P_mtx = CompressMatrix(G, seeds)

        volumes = VolumesMatrix(G)

        reduced_laplacian = P_mtx.transpose() * (sps.diags(laplacian.diagonal(), 0) - laplacian) * P_mtx
        G = nx.from_numpy_matrix((reduced_laplacian - sps.diags(reduced_laplacian.diagonal(), 0)).todense())

        volumes = (P_mtx.transpose() * volumes).todense()
        for node in G.nodes():
            G.node[node]['volume'] = volumes.item(node)
        
        S = AlgebraicMultigrid(G)
        # S = Refine(S)
    return G

def CompressMatrix(G, seeds):
    """Gets the matrix of a coarse graph."""
    global a
    global laplacian
    global neighborhood_size
    global iteration

    nodes_num = len(G.nodes())

    row_array = []
    col_array = []
    data_array = []

    initial = (G.nodes())[0]
    for i in G.nodes():
        if i in seeds:
            if initial != 0:
                row_array.append(i - 1)
            else:
                row_array.append(i)
            col_array.append(seeds.index(i))
            data_array.append(1)
        else:
            neighbors = list(set(G.neighbors(i)).intersection(seeds))
            for j in neighbors:
                if initial != 0:
                    row_array.append(i - 1)
                else:
                    row_array.append(i)
                col_array.append(seeds.index(j))
                edge_sum = 0.0
                for neighbor in neighbors:
                    weight = G.edge[i][neighbor]['weight']
                    if weight >= a:
                        edge_sum += weight
                data_array.append(G.edge[i][j]['weight'] / edge_sum)
            if len(neighbors) > neighborhood_size:
                diff = len(neighbors) - neighborhood_size
                col_array = col_array[:(len(col_array) - diff)]#col

                row_entries = row_array[-len(neighbors):]#row
                row_array = row_array[:(len(row_array) - len(neighbors))]#row
                data_entries = data_array[-len(neighbors):]
                data_array = data_array[:(len(data_array) - len(neighbors))]

                high_entries = []
                high_positions = np.argsort(data_entries)[::-1][:neighborhood_size]
                for pos in high_positions:
                    high_entries.append(data_entries[pos])
                    row_array.append(row_entries[pos])#row
                data_sum = sum(high_entries)
                for entry in high_entries:
                    data_array.append(entry / data_sum)    
    col = np.array(col_array)
    row = np.array(row_array)
    data = np.array(data_array)
    mtx = sps.csr_matrix((data, (row, col)),
        shape=(nodes_num, len(seeds)))
    # print mtx.sum(axis=1)
    # print mtx.A
    # print seeds
    # for summa in mtx.sum(axis=1):
    #    if summa.item(0) > 1.0:
    #        print summa.item(0) - 1.0
    #        print "ROW SUM ERROR!"
    # sys.exit()
    return mtx

def VolumesMatrix(G):
    """Creates and outputs a single column matrix of volumes."""
    row_array = []
    col_array = []
    data_array = []
    for index, node in enumerate(G.nodes(data=True)):
        row_array.append(0)
        col_array.append(index)
        data_array.append(node[1]['volume'])
    row = np.array(row_array)
    col = np.array(col_array)
    data = np.array(data_array)
    mtx = sps.csr_matrix((data, (col, row)),
        shape=(len(G.nodes()), 1))
    return mtx

def FutureVolume(G, nodes):
    """Coarsens the graph G."""

    global Q
    global r
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
    return G


def GetCoarseSeeds(G):
    """Returns a list of nodes that are chosen as seeds for coarse graph."""

    global n
    global Q
    global laplacian

    laplacian = nx.laplacian_matrix(G)
    G = FutureVolume(G, None)
    total_volume = 0.0
    seeds = []
    non_seeds = {}
    for i in nx.nodes_iter(G):
        total_volume += G.node[i]['future_volume']
        non_seeds[i] = G.node[i]['future_volume']
    avg_volume = total_volume / nx.number_of_nodes(G)
    for node, value in non_seeds.items():
        if(value > n * avg_volume):
            seeds.append(node)
            del non_seeds[node]
    G = FutureVolume(G, non_seeds.keys())
    for node in non_seeds.keys():
        non_seeds[node] = G.node[node]['future_volume']
    sorted_nonseeds = sorted(non_seeds.items(), key=operator.itemgetter(1), reverse=True)
    for node, value in sorted_nonseeds:
        seed_edges = 0.0
        total_edges = 0.0
        for neighbor in G.neighbors(node):
            total_edges += G.edge[node][neighbor]['weight']
            if neighbor in seeds:
                seed_edges += G.edge[node][neighbor]['weight']
        if seed_edges / total_edges < Q:
            seeds.append(node)
            del non_seeds[node]
    return seeds

def DefineVariables():
    """Defines the variables for the Algorithm."""

def Solve(G):
    """Solve the graph."""
    global start_time
    
    print "In Solve."
    print "Execution time: " + str(time.time() - start_time) + " seconds"
    return None


def Refine(S):
    """Work back up the V-graph."""
    print "In refine."
    return None
