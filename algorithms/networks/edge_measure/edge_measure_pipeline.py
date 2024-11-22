
from algorithms.networks.edge_measure.cosine import calculate_cosine_similarity
from algorithms.networks.edge_measure.jaccard import calculate_jaccard_similarity
from algorithms.networks.edge_measure.betweenness import calculate_edge_betweenness
from algorithms.networks.edge_measure.dijkstra import calculate_dijkstra_distance

edge_measure_algorithms = {
    "Cosine Similarity": calculate_cosine_similarity,
    "Jaccard Similarity": calculate_jaccard_similarity,
    "Edge Betweenness": calculate_edge_betweenness,
    "Dijkstra Distance": calculate_dijkstra_distance
}

def get_available_edge_measure_algorithms():
    return list(edge_measure_algorithms.keys())

def get_edge_measure_algorithm(algorithm_name):
    return edge_measure_algorithms[algorithm_name]

def edge_measure_pipeline(edge_measure_method, graph, params=None):
    edge_measure_func = get_edge_measure_algorithm(edge_measure_method)
    return edge_measure_pipeline_using_func(edge_measure_func, graph, params)

def edge_measure_pipeline_using_func(edge_measure_func, graph, params=None):
    if params is None:
        params = {}
    return edge_measure_func(graph, params)
