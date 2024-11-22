import time
import numpy as np
from memory_profiler import memory_usage
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from algorithms.clustering import clustering_pipeline, get_available_clustering_algorithms
from algorithms.clustering.clustering_quality_measures import (clustering_quality_measure_pipeline,
                                                               get_available_clustering_quality_measures)
from algorithms.projection import projection_pipeline
from util import load_clustering_projection_data, load_clustering_projection_labels


WHITE = "\033[97m"
BLUE = "\033[34m"
GREEN = "\033[92m"
ORANGE = "\033[38;5;208m"
PINK = "\033[38;5;205m"
RED = "\033[31m"
RESET = "\033[0m"


def test_clustering():
    datasets = [
        {
            "name": "blobs",
            "epsilon": 0.5,
            "min_numbs": 3,
            "n_clusters": 5,
            "affinity": "euclidean",
            "linkage": "ward"
        },
        {
            "name": "iris",
            "epsilon": 1.4,
            "min_numbs": 3,
            "n_clusters": 3,
            "affinity": "euclidean",
            "linkage": "ward"
        },
        {
            "name": "wine",
            "epsilon": 2.3, # Alternatively  "epsilon": 2.9
            "min_numbs": 11, # Alternatively "min_numbs": 21 has better silhouette scores but only two clusters
            "n_clusters": 3,
            "affinity": "euclidean",
            "linkage": "ward"
        }
    ]

    plot_clustering_results(datasets)

    compare_clustering_methods(datasets)


def plot_clustering_results(datasets):

    clustering_algorithms = get_available_clustering_algorithms()

    # Define the number of rows and columns for subplots
    num_datasets = len(datasets)
    num_algorithms = len(clustering_algorithms)
    fig, axes = plt.subplots(num_datasets, num_algorithms + 1, figsize=(5 * (num_algorithms + 1), 5 * num_datasets))
    fig.suptitle("Clustering Results vs. Actual Classes", fontsize=16)

    bar_format = f"{WHITE}⌛  Plotting Clustering Methods...   {{l_bar}}{ORANGE}{{bar}}{WHITE}{{r_bar}}{RESET}"
    with tqdm(total=len(datasets)*len(clustering_algorithms), bar_format=bar_format) as pbar:
        for i, params in enumerate(datasets):
            # Load data and actual labels
            data = load_clustering_projection_data(params['name'])
            actual_labels = load_clustering_projection_labels(params['name'])

            # Apply t-SNE if the data is more than 2-dimensional
            data = np.array(data)
            if data.shape[1] > 2:
                data_2d = projection_pipeline("T-SNE", data)
            else:
                data_2d = data

            # Plot actual classes
            sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1], hue=actual_labels, palette='viridis', ax=axes[i, 0],
                            legend=False)
            axes[i, 0].set_title(f"Actual Classes\n{params['name']}")

            for j, alg in enumerate(clustering_algorithms):
                # Run clustering algorithm
                cluster_labels = clustering_pipeline(alg, data, params)

                # Plot clustering results
                sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1], hue=cluster_labels, palette='viridis', ax=axes[i, j + 1],
                                legend=False)
                axes[i, j + 1].set_title(f"{alg}\n{params['name']}")
                pbar.update(1)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def compare_clustering_methods(datasets):

    # Fetch available quality measures
    quality_measures = get_available_clustering_quality_measures()

    # Initialize dictionary to store cumulative metrics for averaging
    results_table = {alg_name: {measure: 0 for measure in quality_measures}
                     for alg_name in get_available_clustering_algorithms()}

    # Add runtime and memory usage metrics
    for alg_name in results_table:
        results_table[alg_name]["Runtime (s)"] = 0
        results_table[alg_name]["Memory Usage (MiB)"] = 0

    bar_format = f"{WHITE}⌛  Testing Clustering Algorithms...   {{l_bar}}{BLUE}{{bar}}{WHITE}{{r_bar}}{RESET}"
    with tqdm(total=len(datasets) * len(results_table.keys()), bar_format=bar_format) as pbar:
        for j, params in enumerate(datasets):
            points = load_clustering_projection_data(params['name'])
            class_labels = load_clustering_projection_labels(params['name'])

            for alg, metrics in results_table.items():
                # Track memory usage and runtime for the clustering algorithm
                memory_used = memory_usage((clustering_pipeline, (alg, points, params)), max_usage=True)
                start_time = time.time()
                cluster_labels = clustering_pipeline(alg, points, params)
                runtime = time.time() - start_time

                # Calculate metrics
                for measure in quality_measures:
                    value = clustering_quality_measure_pipeline(measure, points, cluster_labels, class_labels)
                    if value is not None and metrics[measure] != "Error":
                        metrics[measure] += value / len(datasets)
                    else:
                        metrics[measure] = "Error"

                metrics["Runtime (s)"] += runtime / len(datasets)
                metrics["Memory Usage (MiB)"] += memory_used / len(datasets)

                pbar.update(1)

    # Print Metric Explanation
    print(f"{WHITE}")
    print("The Algorithms where compared using the following metrics:")
    print("The Jaccard coefficient measures the similarity between the true class labels and the cluster labels. "
          "It ranges from 0 to 1, with 1 indicating a perfect match.")
    print("The Silhouette score measures the quality of the clustering. "
          "It ranges from -1 to 1, with 1 indicating dense, well-separated clusters.")
    print("The Davies-Bouldin index measures the average similarity between each cluster and its most similar cluster. "
          "Lower values indicate better clustering.")
    print("The Purity score measures how well the clusters align. "
          "Purity ranges from 0 to 1, with 1 indicating perfect.")
    print("Runtime and memory usage are also tracked for comparison.")
    print(f"{RESET}", end='')

    # Print results
    for alg_name, metrics in results_table.items():
        print(f"{WHITE}")
        print(f"Results for {alg_name}:")
        for metric_name, value in metrics.items():
            if value == "Error":
                print(f"{metric_name}: {RED}Error (only one cluster was found, calculation not possible){WHITE}")
            elif metric_name == "Runtime (s)":
                print(f"Runtime (ms): {GREEN}{round(value*1000, 4)}{WHITE}")
            else:
                print(f"{metric_name}: {GREEN}{round(value, 4)}{WHITE}")
        print(f"{RESET}")


if __name__ == '__main__':
    test_clustering()