def calculate_jaccard_similarity(graph, _):
    jaccard_similarity = {}

    # Loop through each node and its neighbors
    for node in graph:
        neighbors_node = set(graph[node])
        for neighbor in graph[node]:
            if (node, neighbor) in jaccard_similarity or (neighbor, node) in jaccard_similarity:
                continue  # Avoid duplicate calculation for undirected edges
            neighbors_neighbor = set(graph[neighbor])

            # Calculate Jaccard similarity
            intersection = len(neighbors_node & neighbors_neighbor)
            union = len(neighbors_node | neighbors_neighbor)
            similarity = intersection / union if union != 0 else 0

            # Store the result for the edge (node, neighbor)
            jaccard_similarity[(node, neighbor)] = similarity

    return jaccard_similarity
