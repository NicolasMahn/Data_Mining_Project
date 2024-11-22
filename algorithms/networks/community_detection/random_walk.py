import random
from collections import Counter
import copy
import numpy as np
import itertools


def walktrap_algorithm(graph, params):
    walk_length = params.get("walk_length", 4)
    num_walks = params.get("num_walks", 100)
    num_communities = params.get("n_communities", 3)
    return walktrap(graph, walk_length, num_walks, num_communities)


def walktrap(graph, walk_length, num_walks, num_communities):
    # Step 1: Initialize each node as its own community
    communities = [{node} for node in graph.keys()]
    total_edges = sum(len(neighbors) for neighbors in graph.values()) / 2

    # Perform random walks to adjust communities
    community_labels = {node: idx for idx, node in enumerate(graph.keys())}
    for node in graph:
        walk_communities = Counter()

        # Perform multiple random walks from this node
        for _ in range(num_walks):
            current_node = node
            for _ in range(walk_length):
                if graph[current_node]:
                    current_node = random.choice(graph[current_node])
                    walk_communities[community_labels[current_node]] += 1

        # Assign to the most frequently visited community
        best_community = walk_communities.most_common(1)[0][0]
        community_labels[node] = best_community

    # Merge nodes based on assigned communities
    for node, label in community_labels.items():
        communities[label].add(node)

    # Filter out empty communities and refine the number of communities based on modularity
    communities = [comm for comm in communities if comm]
    while len(communities) > num_communities:
        communities = merge_communities_by_modularity(communities, graph, total_edges)

    return communities


def merge_communities_by_modularity(communities, graph, total_edges):
    possible_new_communities = []
    for i in range(len(communities)):
        for j in range(i + 1, len(communities)):
            new_communities = copy.deepcopy(communities)
            new_communities[j].update(new_communities[i])
            del new_communities[i]
            delta_q = modularity(new_communities, graph, total_edges)
            possible_new_communities.append((delta_q, new_communities))

    # Return the communities with the highest modularity
    return max(possible_new_communities, key=lambda x: x[0])[1]


def modularity(communities, graph, total_edges):
    e = precompute_e(communities, graph, total_edges)
    q = 0
    for i in range(len(communities)):
        e_ii = e[i, i]
        a_i = sum(e[i])
        q += (e_ii - a_i ** 2)
    return q


def precompute_e(communities, graph, total_edges):
    num_communities = len(communities)
    e = np.zeros((num_communities, num_communities))

    for (i, j) in itertools.combinations_with_replacement(range(num_communities), 2):
        e_ij = 0
        for node_i in communities[i]:
            for node_j in communities[j]:
                if node_j in graph[node_i]:
                    e_ij += 1
        e_ij /= (2 * total_edges)
        e[i, j] = e_ij
        e[j, i] = e_ij

    return e
