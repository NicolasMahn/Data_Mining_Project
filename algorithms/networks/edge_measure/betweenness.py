from collections import deque, defaultdict

def calculate_edge_betweenness(graph, _):
    # Initialize betweenness scores for each edge to 0
    edge_betweenness_scores = defaultdict(int)

    # Iterate over each node in the graph
    for source in graph:
        # BFS initialization
        queue = deque([source])
        predecessors = {node: [] for node in graph}  # List of predecessors for each node
        shortest_paths = {node: 0 for node in graph}  # Count of shortest paths to each node
        shortest_paths[source] = 1  # There's one path to the source itself
        distances = {node: -1 for node in graph}  # Distance from source to each node
        distances[source] = 0

        # BFS to calculate shortest paths and distances
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                # Found the node for the first time
                if distances[neighbor] == -1:
                    queue.append(neighbor)
                    distances[neighbor] = distances[node] + 1
                # Count the shortest paths to each neighbor
                if distances[neighbor] == distances[node] + 1:
                    shortest_paths[neighbor] += shortest_paths[node]
                    predecessors[neighbor].append(node)

        # Calculate the betweenness for each edge using dependency accumulation
        dependencies = {node: 0 for node in graph}
        nodes_by_distance = sorted(distances.keys(), key=lambda x: -distances[x])

        for node in nodes_by_distance:
            for pred in predecessors[node]:
                # Calculate the fraction of paths through this predecessor
                weight = (shortest_paths[pred] / shortest_paths[node]) * (1 + dependencies[node])
                # Add weight to the edge (pred, node)
                edge_betweenness_scores[(min(pred, node), max(pred, node))] += weight
                dependencies[pred] += weight

    # Normalize the edge betweenness scores by dividing by 2
    for edge in edge_betweenness_scores:
        edge_betweenness_scores[edge] /= 2

    return edge_betweenness_scores
