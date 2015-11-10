import sys, time, random, operator, json
import networkx as nx
import scipy.sparse as sps
# import scipy
import numpy as np
import matplotlib.pyplot as plt

# volume = None
complete_time = None
start_time = None
iteration = 0

__all__ = ['AlgebraicMultigrid']


def AlgebraicMultigrid(G, **args):
    """Accept a graph and attempt to solve it."""

    global complete_time
    global start_time
    global iteration

    node_limit = 10
    edge_weight_limit = 0.0

    # if iteration == 1:
    complete_time = time.time()
    print "Iteration: " + str(iteration)
    print "Nodes: " + str(len(G.nodes()))
    print "Edges: " + str(len(G.edges()))

    # DrawGraph(G)
    # print G.nodes(data=True)

    if nx.number_of_nodes(G) <= node_limit:
        G = Solve(G)
    else:
        start_time = time.time()

        volumes = VolumesMatrix(G)
        # print "Generating Volumes: " + str(time.time() - start_time) + " seconds"
        # start_time = time.time()

        # laplacian = sps.csr_matrix(nx.laplacian_matrix(G, weight='weight'))
        # laplacian = nx.laplacian_matrix(G, weight='weight')

        seeds = GetCoarseSeeds(G)
        # print seeds

        with open('iterations_schema.json', 'r') as schema_file:
            schema = json.load(schema_file)

        schema["iterations"][iteration] = {}
        # naming[iteration] = {}
        for index, seed in enumerate(seeds):
            schema["iterations"][iteration][index] = seed

        with open("iterations_schema.json", 'w') as outfile:
            json.dump(schema, outfile)

        # sys.exit()
        # print "Getting seeds: " + str(time.time() - start_time) + " seconds"
        # start_time = time.time()

        #ScaleEdges(G, edge_weight_limit, seeds)
        # print "Scaling Edges: " + str(time.time() - start_time) + " seconds"
        # start_time = time.time()

        P_mtx = CompressMatrix(G, seeds, edge_weight_limit)
        # print "Compressing Matrix: " + str(time.time() - start_time) + " seconds"
        # start_time = time.time()

        reduced_laplacian = P_mtx.transpose() * nx.adjacency_matrix(G, weight='weight') * P_mtx
        G = nx.from_numpy_matrix((reduced_laplacian - sps.diags(reduced_laplacian.diagonal(), 0)).todense())
        # print "Creating Coarse Graph: " + str(time.time() - start_time) + " seconds"
        # start_time = time.time()

        volumes = (P_mtx.transpose() * volumes).todense()
        for node in G.nodes():
            G.node[node]['volume'] = volumes.item(node)
        # print "Distributing Volumes: " + str(time.time() - start_time) + " seconds"
        # sys.exit()
        iteration += 1
        S = AlgebraicMultigrid(G)
        # S = Refine(S)
    return G

def DrawGraph(Graph):
    global iteration    

    # layout = nx.random_layout(Graph, dim=2)
    layout = nx.spring_layout(Graph, iterations=200)
    nx.draw(Graph, node_size=1, pos=layout, font_size=.3, font_color="blue",# layout="sfdp",
        # node_color='#A0CBE2', edge_color='#BB0000', 
        width=.2, linewidths=.2, edge_cmap=plt.cm.Blues, with_labels=True
        )
    plt.savefig("plots/start_graph" + str(iteration) + ".pdf", # dpi=1500, facecolor='w', edgecolor='w', 
        format="PDF", # format="None", orientation='portrait', papertype=None, 
        transparent=False, bbox_inches=None, pad_inches=0.1
        )
    plt.hold(False)


def CompressMatrix(G, seeds, edge_weight_limit):
    """Gets the matrix of a coarse graph."""

    neighborhood_size = 2

    nodes_num = len(G.nodes())

    row_array = []
    col_array = []
    data_array = []

    for i in G.nodes():
        if i in seeds:
            row_array.append(i)
            col_array.append(seeds.index(i))
            data_array.append(1)
        else:
            neighbors = list(set(G.neighbors(i)).intersection(seeds))
            for j in neighbors:
                row_array.append(i)
                col_array.append(seeds.index(j))
                edge_sum = 0.0
                for neighbor in neighbors:
                    weight = G.edge[i][neighbor]['weight']
                    if weight >= edge_weight_limit:
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

    for i in nodes:
        G.node[i]['future_volume'] = G.node[i]['volume']
        for j in list(set(G.neighbors(i)).intersection(nodes)):
            degree = G.degree(j)
            adjacency = degree / min(r, Q * degree)
            norm_weight = G.edge[i][j]['weight'] / G.node[j]['sum_weight']
            G.node[i]['future_volume'] += G.node[j]['volume'] * min(1.0, adjacency * norm_weight)


def GetCoarseSeeds(G):
    """Returns a list of nodes that are chosen as seeds for coarse graph."""

    global complete_time

    n = 2.0
    Q = .5

    nodes = G.nodes()
    seeds = []
    non_seeds = []

    run_time = time.time()

    for k in nodes:
        G.node[k]['sum_weight'] = sum(edge[2]['weight'] for edge in G.edges_iter(k, data=True))

    FutureVolume(G, nodes, Q)

    total_volume = sum((nx.get_node_attributes(G, 'future_volume')).values())
    avg_volume = total_volume / nx.number_of_nodes(G)
    for node, data in G.nodes(data=True):
        if data['future_volume'] > n * avg_volume:
            seeds.append(node)
    non_seeds = set(nodes).difference(seeds)
    FutureVolume(G, non_seeds, Q)
    non_seeds = sorted(non_seeds, key=lambda node: G.node[node]['future_volume'], reverse=True)

    for node in non_seeds:
        seed_sum = 0.0
        seed_edges = set(seeds).intersection(G.neighbors(node))
        for seed in seed_edges:
            seed_sum += G[node][seed]['weight']
        if seed_sum / G.node[node]['sum_weight'] <= Q: # [ <= Q ] this difference causes division by zero in Edge Scaling
            seeds.append(node)
    return seeds


def ScaleEdges(G, edge_weight_limit, seeds):
    """Scales distance between strongly connected nodes to avoid second handed pull of their aggregates."""
    initializations = 10
    iterations = 10
    omega = 0.5
    nodes = G.nodes(data=True)

    scales = np.zeros((initializations, len(nodes)))
    # minimum = (np.random.rand(1, len(nodes))).flatten() - .5
    # maximum = np.copy(minimum)
    # initial = np.copy(minimum)
    # previous = np.copy(minimum)
    for init in range(initializations):
        initial = (np.random.rand(1, len(nodes))).flatten() - .5
        minimum = (np.full((1, len(nodes)), 1))[0]
        maximum = (np.full((1, len(nodes)), -1))[0]
        previous = np.copy(initial)
        for iteration in range(iterations):
            current = np.zeros(len(nodes))
            for node, data in nodes:
                weights = data['sum_weight']
                scale_neighbors = 0.0
                for edge in G.edges(node, data=True):
                    scale_neighbors += edge[2]['weight'] * previous[edge[1]]
                scale = omega * previous[node] + (1.0 - omega) * (scale_neighbors / weights)
                current[node] = scale
                if scale < minimum[node]:
                    minimum[node] = scale
                if scale > maximum[node]:
                    maximum[node] = scale
            previous = np.copy(current)
        for index, center in enumerate(initial):
            a = abs(center - minimum[index])
            b = abs(maximum[index] - center)
            scales[init][index] = a / (a + b)
    for node, neighbor in G.edges():
        summa = 0.0
        for init in scales:
            summa += (init[node] - init[neighbor]) ** 2
        G[node][neighbor]['weight'] = 1.0 / summa


def DefineVariables():
    """Defines the variables for the Algorithm."""


def Solve(G):
    """Solve the graph."""
    global complete_time
    global iteration

    decryptor = iteration - 1

    print "In Solve."
    # print "Execution time: " + str(time.time() - complete_time) + " seconds"
    print G.nodes(data=True)
    print G.edges(data=True)

    with open('iterations_schema.json', 'r') as schema_file:
            schema = json.load(schema_file)
    with open('name_schema.json', 'r') as schema_file:
            names = json.load(schema_file)
    # print schema
    # print schema["iterations"]
    for node in G.nodes():
        translate = node
        schema_trace = str(node)
        while decryptor >= 0:
            translate = str(schema["iterations"][str(decryptor)][str(translate)])
            schema_trace += " -> " + translate
            decryptor -= 1
        for table in names:
            if names[table]["id"] == int(translate):
                schema_trace += " -> " + table
        decryptor = iteration - 1
        print schema_trace
    return None


def Refine(S):
    """Work back up the V-graph."""
    print "In refine."
    return None
