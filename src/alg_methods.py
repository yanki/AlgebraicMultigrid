import networkx as nx
import settings

Q = settings.threshhold
r = settings.neighborhood_size


def AlgebraicMultigrid(G):
    """Accept a graph and attempt to solve it."""
    if nx.number_of_nodes(G) <= 7:
        S = Solve(G)
    else:
        G = FutureVolume(G)
        print G.nodes(data=True)
        print G.edges(data=True)
        """
        1. find algebraic distance
        2. future volume
        3. coarse nodes
        4. coarse edges
        """
        # S = AlgebraicMultigrid(G)
        # S = Refine(S)
        S = None
    return S


def FutureVolume(G):
    """Coarsens the graph G."""

    global Q
    global r

    laplacian = nx.laplacian_matrix(G)
    for i in nx.nodes_iter(G):
        G.node[i]['future_volume'] = G.node[i]['volume']
        for j in nx.all_neighbors(G, i):
            degree = G.degree(j)
            adjacency = degree / min(r, Q * degree)
            sum_weight = 0.0
            for k in (k for k in nx.all_neighbors(G, i) if laplacian[i - 1, k - 1] < 0.0):
                sum_weight += G.edge[i][k]['weight']
            norm_weight = G.edge[i][j]['weight'] / sum_weight
            G.node[i]['future_volume'] += G.node[j]['volume'] * min(1.0, adjacency * norm_weight)
    print "Calculated Future Volume."
    return G


def Solve(G):
    """Solve the graph."""
    print "In Solve."
    return None


def Refine(S):
    """Work back up the V-graph."""
    print "In refine."
    return None
