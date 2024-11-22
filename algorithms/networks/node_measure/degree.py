

def calculate_degree_centrality(graph, _):
    # Degree centrality is simply the count of connections for each node
    degree_centrality = {node: len(neighbors) for node, neighbors in graph.items()}
    return degree_centrality
