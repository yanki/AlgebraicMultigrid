import networkx as nx
# import scipy as sci
import settings

Q = settings.threshhold
r = settings.neighborhood_size


def AlgebraicMultigrid(G):
    """Accept a graph and attempt to solve it."""
    # print nx.number_of_nodes(G)
    # print nx.number_of_edges(G)
    if nx.number_of_nodes(G) <= 7:
        S = Solve(G)
    else:
        G = Coarsen(G)
        # S = AlgebraicMultigrid(G)
        # S = Refine(S)
        S = None
    return S


def Coarsen(G):
    """Coarsens the graph G."""

    global Q
    global r

    laplacian = nx.laplacian_matrix(G)
    print laplacian
    # print G.adj
    volume_sum = 0
    count = 0
    for iteration, i in enumerate(nx.nodes_iter(G)):
        G.node[i]['future_volume'] = G.node[i]['volume']
        for j in nx.all_neighbors(G, i):
            degree = G.degree(j)
            adjacency = degree / min(r, Q * degree)
            sum_weight = 0.0
            for k in (k for k in nx.all_neighbors(G, i) if laplacian[i - 1, k - 1] < 0.0):
                sum_weight = sum_weight + G.edge[i][k]['weight']
            norm_weight = G.edge[i][j]['weight'] / sum_weight
            G.node[i]['future_volume'] = G.node[i]['future_volume'] + G.node[j]['volume'] * min(1.0, adjacency * norm_weight)
        volume_sum = volume_sum + G.node[i]['future_volume']
        count = iteration + 1
    avg_volume = volume_sum / count
    print "avg_volume: " + str(avg_volume) + " count: " + str(count)
    print G.nodes(data=True)
    print G.edges(data=True)
    """
    1. calculate future volume
    2. algebraic distance between nodes
    3. find coarse seeds for aggregates
    4. find corase edges
    5. filtering
    """
    print "Coarsening..."
    return None


def Solve(G):
    """Solve the graph."""
    print "In Solve."
    return None


def Refine(S):
    """Work back up the V-graph."""
    print "In refine."
    return None
