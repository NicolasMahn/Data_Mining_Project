import random
import numpy as np
from networkx.algorithms.threshold import threshold_graph


def louvain_algorithm(graph, params):
    max_iter = params.get("max_iterations", 10000)

    # Step 1: Initialize each node as its own community
    community = {node: node for node in graph}
    improvement = True

    # Continue merging communities while there are modularity gains
    while improvement and max_iter > 0:
        improvement = False
        for node in graph:
            node_community = community[node]
            best_community = node_community
            best_gain = 0

            # Calculate modularity gain for moving node to each neighboring community
            neighbor_communities = set(community[neighbor] for neighbor in graph[node])
            for neighbor_community in neighbor_communities:
                community[node] = neighbor_community  # Temporarily assign to test gain
                gain = modularity_gain(graph, community)
                if gain > best_gain:
                    best_gain = gain
                    best_community = neighbor_community

            # Reassign the node to the community with the highest modularity gain
            if best_community != node_community:
                community[node] = best_community
                improvement = True
        max_iter -= 1

    # Group nodes into communities based on final labels
    communities = {}
    for node, comm in community.items():
        if comm not in communities:
            communities[comm] = set()
        communities[comm].add(node)

    return list(communities.values())


def modularity_gain(graph, community):
    m = sum(len(neighbors) for neighbors in graph.values()) / 2  # Total number of edges
    modularity = 0.0

    # Calculate modularity based on community assignments
    for node, neighbors in graph.items():
        node_community = community[node]
        for neighbor in neighbors:
            neighbor_community = community[neighbor]
            if node_community == neighbor_community:
                modularity += 1 - (len(graph[node]) * len(graph[neighbor])) / (2 * m)
            else:
                modularity -= (len(graph[node]) * len(graph[neighbor])) / (2 * m)

    return modularity / (2 * m)
