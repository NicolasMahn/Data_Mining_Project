from collections import deque


def calculate_betweenness_centrality(graph, _):
    betweenness_centrality = {node: 0.0 for node in graph}  # Initialize centrality values

    for s in graph:  # For each node as the source node
        # Step 1: Initialize structures
        stack = []
        pred = {w: [] for w in graph}  # Predecessors
        sigma = {t: 0 for t in graph}  # Number of shortest paths from s to t
        dist = {t: -1 for t in graph}  # Distance from source to node
        sigma[s] = 1
        dist[s] = 0

        # Step 2: Breadth-first search
        queue = deque([s])
        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in graph[v]:
                # Path discovery
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                # Path counting
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        # Step 3: Accumulation phase
        delta = {t: 0 for t in graph}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                betweenness_centrality[w] += delta[w]

    # Step 4: Normalize (optional, for undirected graphs)
    for node in betweenness_centrality:
        betweenness_centrality[node] /= 2.0  # Divide by 2 for undirected graphs

    return betweenness_centrality
