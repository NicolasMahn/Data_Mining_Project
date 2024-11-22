from collections import deque


def calculate_closeness_centrality(graph, _):
    closeness_centrality = {}

    for node in graph:
        # Calculate the shortest path distances from the node to all other nodes using BFS
        distances = {n: float('inf') for n in graph}
        distances[node] = 0
        queue = deque([node])

        while queue:
            current = queue.popleft()
            for neighbor in graph[current]:
                if distances[neighbor] == float('inf'):  # Not visited
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

        # Sum of distances to all other nodes
        reachable_nodes = [d for d in distances.values() if d < float('inf')]
        if len(reachable_nodes) > 1:  # Exclude isolated nodes
            closeness_centrality[node] = (len(reachable_nodes) - 1) / sum(reachable_nodes)
        else:
            closeness_centrality[node] = 0.0  # Isolated node or no reachable nodes

    return closeness_centrality
