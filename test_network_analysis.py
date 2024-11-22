import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

from algorithms.networks.community_detection import (community_detection_pipeline,
                                                    get_available_community_detection_algorithms)
from algorithms.networks.edge_measure import get_available_edge_measure_algorithms, edge_measure_pipeline
from algorithms.networks.node_measure import get_available_node_measure_algorithms, node_measure_pipeline
from util import load_test_network

WHITE = "\033[97m"
BLUE = "\033[34m"
GREEN = "\033[92m"
ORANGE = "\033[38;5;208m"
PINK = "\033[38;5;205m"
RED = "\033[31m"
RESET = "\033[0m"

def test_community_detection():
    datasets = [
        {
            "name": "karate",
            "n_communities": 2,
            "start_node": 0
        },
        {
            "name": "three_communities",
            "n_communities": 3,
            "start_node": 0
        },
        {
            "name": "les_miserables",
            "n_communities": 5,
            "start_node": 0
        }
    ]

    plot_community_detection_results(datasets)

    compare_node_measures(datasets)

    compare_edge_measures(datasets)


def plot_community_detection_results(datasets):
    community_detection_algorithms = get_available_community_detection_algorithms()

    # Define the number of rows and columns for subplots
    num_datasets = len(datasets)
    num_algorithms = len(community_detection_algorithms)
    fig, axes = plt.subplots(num_datasets, num_algorithms, figsize=(5 * num_algorithms, 5 * num_datasets))
    fig.suptitle("Community Detection Results vs. Actual Communities", fontsize=16)

    bar_format = f"{WHITE}⌛  Plotting Projection Methods...   {{l_bar}}{ORANGE}{{bar}}{WHITE}{{r_bar}}{RESET}"
    with tqdm(total=len(datasets)*len(community_detection_algorithms), bar_format=bar_format) as pbar:
        for i, params in enumerate(datasets):
            # Load graph and actual communities
            graph, _ = load_test_network(params['name'])

            for j, alg in enumerate(community_detection_algorithms):
                # Run community detection algorithm
                communities = community_detection_pipeline(alg, graph, params)

                # Plot detected communities
                visualize_communities(graph, communities, ax=axes[i, j])
                axes[i, j].set_title(f"{alg}\n{params['name']}")
                pbar.update(1)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def visualize_communities(graph, communities, labels=None, ax=None):
    G = nx.Graph()
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    pos = nx.spring_layout(G)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightpink', 'lightyellow', 'lightgray', 'lightcyan']

    for i, community in enumerate(communities):
        color = colors[i % len(colors)]
        for node in community:
            G.nodes[node]['color'] = color

    if labels is not None:
        for label, node in labels.items():
            G.nodes[node]['label'] = label

        nx.draw(G, pos, node_color=[G.nodes[node]['color'] for node in G.nodes], with_labels=True, edge_color='gray',
                node_size=500, font_size=10, labels={node: G.nodes[node].get('label', '') for node in G.nodes}, ax=ax)

    else:
        nx.draw(G, pos, node_color=[G.nodes[node]['color'] for node in G.nodes], with_labels=False, edge_color='gray',
                node_size=500, ax=ax)


def compare_node_measures(datasets):
    node_measure_algorithms = get_available_node_measure_algorithms()

    for params in datasets:
        # Load graph and actual communities
        graph, _ = load_test_network(params['name'])

        print(f"\n{WHITE}Dataset: {params['name']}")
        for alg in node_measure_algorithms:
            # Run community detection algorithm
            node_measure = node_measure_pipeline(alg, graph, params)

            print(f"\n{WHITE}Node measure algorithm: {alg}")
            print(f"{GREEN}{node_measure}")
            print(RESET)

def compare_edge_measures(datasets):
    edge_measure_algorithms = get_available_edge_measure_algorithms()

    for params in datasets:
        # Load graph and actual communities
        graph, _ = load_test_network(params['name'])

        print(f"\n{WHITE}Dataset: {params['name']}")
        for alg in edge_measure_algorithms:
            # Run community detection algorithm
            node_measure = edge_measure_pipeline(alg, graph, params)

            print(f"\n{WHITE}Node measure algorithm: {alg}")
            print(f"{GREEN}{node_measure}")
            print(RESET)


if __name__ == "__main__":
    test_community_detection()