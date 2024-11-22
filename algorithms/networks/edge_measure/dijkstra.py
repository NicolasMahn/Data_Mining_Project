import heapq


def calculate_dijkstra_distance(graph, params):
    start_node = params.get("start_node", list(graph.keys())[0])

    # Initialize the priority queue
    priority_queue = [(0, start_node)]  # (distance, node)
    shortest_paths = {node: float('inf') for node in graph}  # Set initial distances to infinity
    shortest_paths[start_node] = 0  # Distance to the start node is 0
    visited = set()  # Track visited nodes

    # Process nodes in the priority queue
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_node in visited:
            continue
        visited.add(current_node)

        # Explore neighbors of the current node
        for neighbor in graph[current_node]:
            weight = 1  # Assume unweighted graph
            distance = current_distance + weight

            # Only update if a shorter path is found
            if distance < shortest_paths[neighbor]:
                shortest_paths[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return shortest_paths
