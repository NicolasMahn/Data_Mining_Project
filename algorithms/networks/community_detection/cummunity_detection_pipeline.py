
from algorithms.networks.community_detection.fast_newman import fast_newman_algorithm
from algorithms.networks.community_detection.louvain import louvain_algorithm
from algorithms.networks.community_detection.random_walk import walktrap_algorithm

community_detection_algorithms = {
    "Fast Newman": fast_newman_algorithm,
    "Louvain": louvain_algorithm,
    "Random Walk": walktrap_algorithm
}

def get_available_community_detection_algorithms():
    return list(community_detection_algorithms.keys())

def get_community_detection_algorithm(algorithm_name):
    return community_detection_algorithms[algorithm_name]

def community_detection_pipeline(community_detection_method, data, params=None):
    community_detection_func = get_community_detection_algorithm(community_detection_method)
    return community_detection_pipeline_using_func(community_detection_func, data, params)

def community_detection_pipeline_using_func(community_detection_func, data, params=None):
    if params is None:
        params = {}
    return community_detection_func(data, params)
