import itertools
import numpy as np
import copy

def fast_newman_algorithm(graph, params):
    num_communities = params.get("n_communities", 3)
    return fast_newman(graph, num_communities)

def fast_newman(graph, num_communities):
    communities = [{node} for node in graph.keys()]
    total_edges = sum(len(neighbors) for neighbors in graph.values()) / 2

    while len(communities) > num_communities:
        possible_new_communities = []

        for i in range(len(communities)):
            for j in range(len(communities)):
                if i != j:
                    new_communities = copy.deepcopy(communities)
                    new_communities[j].update(new_communities[i])
                    del new_communities[i]
                    delta_q = modularity(new_communities, graph, total_edges)
                    possible_new_communities.append((delta_q, new_communities))

        communities = max(possible_new_communities, key=lambda x: x[0])[1]
    return communities


def precompute_e(communities, graph, total_edges):
    num_communities = len(communities)
    e = np.zeros((num_communities, num_communities))

    # Iterate over all unique pairs of communities (i, j) with i <= j
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

def modularity(communities, graph, total_edges):
    # Q = \sum_i (e_{ii} - a_i^2)
    # e_{ij} = fraction of edges that connect nodes in community i to nodes in community j
    # a_i = \sum_j e_{ij}
    # e_{ii} = fraction of edges that connect nodes in community i to nodes in community i

    e = precompute_e(communities, graph, total_edges)

    q = 0
    for i in range(len(communities)):
        e_ii = e[i, i]
        a_i = sum(e[i])

        q += (e_ii - a_i ** 2)

    return q