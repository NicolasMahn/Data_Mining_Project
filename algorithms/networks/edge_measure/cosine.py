import math


def calculate_cosine_similarity(graph, _):
    cosine_similarity = {}

    # Loop through each node and its neighbors
    for node in graph:
        neighbors_node = set(graph[node])
        for neighbor in graph[node]:
            if (node, neighbor) in cosine_similarity or (neighbor, node) in cosine_similarity:
                continue  # Avoid duplicate calculation for undirected edges

            neighbors_neighbor = set(graph[neighbor])

            # Calculate dot product and magnitudes
            intersection = neighbors_node & neighbors_neighbor
            dot_product = len(intersection)
            norm_node = math.sqrt(len(neighbors_node))
            norm_neighbor = math.sqrt(len(neighbors_neighbor))

            # Calculate cosine similarity
            similarity = dot_product / (norm_node * norm_neighbor) if norm_node * norm_neighbor != 0 else 0

            # Store the result for the edge (node, neighbor)
            cosine_similarity[(node, neighbor)] = similarity

    return cosine_similarity
