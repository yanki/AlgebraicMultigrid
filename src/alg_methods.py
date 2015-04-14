import sys
import time
import networkx as nx
import scipy.sparse as sps
# import scipy
import operator
import numpy as np
import matplotlib.pyplot as plt

# volume = None
complete_time = None
start_time = None
iteration = 1

__all__ = ['AlgebraicMultigrid']


def AlgebraicMultigrid(G, **args):
    """Accept a graph and attempt to solve it."""

    global complete_time
    global start_time
    global iteration

    node_limit = 8

    # if iteration == 1:
    complete_time = time.time()
    print "Iteration: " + str(iteration)
    print "Nodes: " + str(len(G.nodes()))
    print "Edges: " + str(len(G.edges()))

    # nx.draw(G, node_size=50, layout="sfdp")
    # plt.savefig("plots/graph" + str(iteration) + ".pdf", format="PNG")
    # plt.show()

    if nx.number_of_nodes(G) <= node_limit:
        G = Solve(G)
    else:
        start_time = time.time()

        volumes = VolumesMatrix(G)
        print "Generating Volumes: " + str(time.time() - start_time) + " seconds"
        start_time = time.time()

        seeds, laplacian = GetCoarseSeeds(G)
        # print G.nodes(data=True)
        print "Getting seeds: " + str(time.time() - start_time) + " seconds"
        start_time = time.time()
        # if iteration == 2:
        #     sys.exit()

        P_mtx = CompressMatrix(G, seeds, laplacian)
        # print P_mtx.A
        print "Compressing Matrix: " + str(time.time() - start_time) + " seconds"
        start_time = time.time()

        reduced_laplacian = P_mtx.transpose() * (sps.diags(laplacian.diagonal(), 0) - laplacian) * P_mtx
        G = nx.from_numpy_matrix((reduced_laplacian - sps.diags(reduced_laplacian.diagonal(), 0)).todense())
        print "Creating Coarse Graph: " + str(time.time() - start_time) + " seconds"
        start_time = time.time()

        volumes = (P_mtx.transpose() * volumes).todense()
        for node in G.nodes():
            G.node[node]['volume'] = volumes.item(node)
        print "Distributing Volumes: " + str(time.time() - start_time) + " seconds"
        # sys.exit()
        iteration += 1
        S = AlgebraicMultigrid(G)
        # S = Refine(S)
    return G


def CompressMatrix(G, seeds, laplacian):
    """Gets the matrix of a coarse graph."""
    a = 0.0

    neighborhood_size = 6

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
                col_array = col_array[:(len(col_array) - diff)]

                row_entries = row_array[-len(neighbors):]
                row_array = row_array[:(len(row_array) - len(neighbors))]
                data_entries = data_array[-len(neighbors):]
                data_array = data_array[:(len(data_array) - len(neighbors))]

                high_entries = []
                high_positions = np.argsort(data_entries)[::-1][:neighborhood_size]
                for pos in high_positions:
                    high_entries.append(data_entries[pos])
                    row_array.append(row_entries[pos])
                data_sum = sum(high_entries)
                for entry in high_entries:
                    data_array.append(entry / data_sum)
    col = np.array(col_array)
    row = np.array(row_array)
    data = np.array(data_array)
    mtx = sps.csr_matrix((data, (row, col)),
        shape=(nodes_num, len(seeds)))
    # print mtx
    # print mtx.sum(axis=1)
    # print mtx.A
    # print seeds
    # for summa in mtx.sum(axis=1):
    #     if summa.item(0) > 1.0:
    #         print summa.item(0) - 1.0
    #         print "ROW SUM ERROR!"
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


def FutureVolume(G, nodes, Q):
    """Coarsens the graph G."""

    r = 2.0

    previous = None

    for i in nodes:
        G.node[i]['future_volume'] = G.node[i]['volume']
        for j in list(set(G.neighbors(i)).intersection(nodes)):
            degree = G.degree(j)
            adjacency = degree / min(r, Q * degree)
            sum_weight = 0.0
            for k in G.edges_iter(j, data=True):
                sum_weight += k[2]['weight']
            norm_weight = G.edge[i][j]['weight'] / sum_weight
            G.node[i]['future_volume'] += G.node[j]['volume'] * min(1.0, adjacency * norm_weight)
    return G


def GetCoarseSeeds(G):
    """Returns a list of nodes that are chosen as seeds for coarse graph."""

    global complete_time

    n = 2.0
    Q = .5

    laplacian = nx.laplacian_matrix(G, weight='weight')
    G = FutureVolume(G, G.nodes(), Q)
    seeds = []
    total_volume = sum((nx.get_node_attributes(G, 'future_volume')).values())
    avg_volume = total_volume / nx.number_of_nodes(G)
    for node, data in G.nodes(data=True):
        if data['future_volume'] > n * avg_volume:
            seeds.append(node)
    non_seeds = set(G.nodes()).difference(seeds)
    G = FutureVolume(G, non_seeds, Q)
    non_seeds = sorted(non_seeds, key=lambda node: G.node[node]['future_volume'], reverse=True)
    for node in non_seeds:
        seed_edges = 0.0
        total_edges = 0.0
        for data in G.edges_iter(node, data=True):
            total_edges += data[2]['weight']
            if data[1] in seeds:
                seed_edges += data[2]['weight']
        if seed_edges / total_edges < Q:
            seeds.append(node)
    return seeds, laplacian


def DefineVariables():
    """Defines the variables for the Algorithm."""


def Solve(G):
    """Solve the graph."""
    global complete_time

    print "In Solve."
    print "Execution time: " + str(time.time() - complete_time) + " seconds"
    return None


def Refine(S):
    """Work back up the V-graph."""
    print "In refine."
    return None
