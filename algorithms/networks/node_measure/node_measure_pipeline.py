from algorithms.networks.node_measure.degree import calculate_degree_centrality
from algorithms.networks.node_measure.closeness import calculate_closeness_centrality
from algorithms.networks.node_measure.betweenness import calculate_betweenness_centrality
from algorithms.networks.node_measure.page_rank import calculate_pagerank

node_measure_algorithms = {
    "Degree Centrality": calculate_degree_centrality,
    "Closeness Centrality": calculate_closeness_centrality,
    "Betweenness Centrality": calculate_betweenness_centrality,
    "PageRank": calculate_pagerank
}

def get_available_node_measure_algorithms():
    return list(node_measure_algorithms.keys())

def get_node_measure_algorithm(algorithm_name):
    return node_measure_algorithms[algorithm_name]

def node_measure_pipeline(node_measure_method, graph, params=None):
    node_measure_func = get_node_measure_algorithm(node_measure_method)
    return node_measure_pipeline_using_func(node_measure_func, graph, params)

def node_measure_pipeline_using_func(node_measure_func, graph, params=None):
    if params is None:
        params = {}
    return node_measure_func(graph, params)
