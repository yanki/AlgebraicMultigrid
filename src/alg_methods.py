import networkx as nx


def AlgebraicMultigrid(G):
    """Accept a graph and attempt to solve it."""
    # print nx.number_of_nodes(G)
    # print nx.number_of_edges(G)
    if nx.number_of_nodes(G) <= 8:
        S = Solve(G)
    else:
        G = Coarsen(G)
        S = AlgebraicMultigrid(G)
        # S = Refine(S)
    return S


def Coarsen(G):
    """Coarsens the graph G."""
    """
    1. calculate future volume (mine)
    2. algebraic distance between nodes (separate paper; tiffany's)
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
