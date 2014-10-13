import networkx as nx
import settings

Q = settings.threshhold
r = settings.neighborhood_size
n = settings.aggregate_size
laplacian = None

__all__ = ['AlgebraicMultigrid']

def AlgebraicMultigrid(G):
    """Accept a graph and attempt to solve it."""
    global n
    global laplacian

    if nx.number_of_nodes(G) <= 7:
        G = Solve(G)
    else:
        laplacian = nx.laplacian_matrix(G)
        G = FutureVolume(G, None)
        # print G.nodes(data=True)
        # print G.edges(data=True)
        total_volume = 0.0
        seeds = {}
        non_seeds = {}
        for i in nx.nodes_iter(G):
            total_volume += G.node[i]['future_volume']
            non_seeds[i] = G.node[i]['future_volume']
        avg_volume = total_volume / nx.number_of_nodes(G)
        print "Average future volume: " + str(avg_volume) + " Number of nodes: " + str(nx.number_of_nodes(G))
        print "Incremented by n: " + str(n * avg_volume)
        for node, value in non_seeds.items():
            if(value > n * avg_volume):
                seeds[node] = value
                del non_seeds[node]
        print non_seeds
        print seeds
        # G = FutureVolume(G, non_seeds.keys())
        # print non_seeds
        # print seeds
        """
        1. find algebraic distance
        2. future volume
        3. coarse nodes
        4. coarse edges
        """
        # S = AlgebraicMultigrid(G)
        # S = Refine(S)
    return G


def FutureVolume(G, nodes):
    """Coarsens the graph G."""

    global Q
    global r
    global laplacian

    if(nodes is None):
        node_list = G.nodes()
    else:
        node_list = nodes
        print node_list
    for i in node_list:
        G.node[i]['future_volume'] = G.node[i]['volume']
        for j in G.neighbors(i):
            degree = G.degree(j)
            adjacency = degree / min(r, Q * degree)
            sum_weight = 0.0
            for k in (k for k in G.neighbors(i) if laplacian[i - 1, k - 1] < 0.0):
                sum_weight += G.edge[i][k]['weight']
            norm_weight = G.edge[i][j]['weight'] / sum_weight
            G.node[i]['future_volume'] += G.node[j]['volume'] * min(1.0, adjacency * norm_weight)
    print "Calculated Future Volume."
    return G

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
