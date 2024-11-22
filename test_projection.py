import math

import numpy as np
import argparse
import time
from memory_profiler import memory_usage
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from algorithms.projection import projection_pipeline, get_available_projection_algorithms
from algorithms.projection.projection_quality_measures.quality_measures_pipeline import (
    projection_quality_measure_pipeline, get_available_projection_quality_measures)
from util import load_clustering_projection_data, load_clustering_projection_labels

WHITE = "\033[97m"
BLUE = "\033[34m"
GREEN = "\033[32m"
ORANGE = "\033[38;5;208m"
PINK = "\033[38;5;205m"
RED = "\033[31m"
RESET = "\033[0m"

def test_projection():
    datasets = [
        {
            "name": "blobs",
        },
        {
            "name": "iris",
        },
        {
            "name": "wine",
        }
    ]

    plot_projection_results(datasets)

    compare_projection_methods(datasets)

def plot_projection_results(datasets):
    projection_methods = get_available_projection_algorithms()

    # Define the number of rows and columns for subplots
    num_datasets = len(datasets)
    num_projections = len(projection_methods)
    fig, axes = plt.subplots(num_datasets, num_projections, figsize=(5 * num_projections, 5 * num_datasets))
    fig.suptitle("Projection Results vs. Actual Classes", fontsize=16)

    bar_format = f"{WHITE}⌛  Plotting Projection Methods...   {{l_bar}}{ORANGE}{{bar}}{WHITE}{{r_bar}}{RESET}"
    with tqdm(total=len(datasets)*len(projection_methods), bar_format=bar_format) as pbar:
        for i, params in enumerate(datasets):
            # Load data and actual labels
            data = load_clustering_projection_data(params['name'])
            actual_labels = load_clustering_projection_labels(params['name'])

            for j, method in enumerate(projection_methods):
                # Apply projection method
                data_2d = projection_pipeline(method, data)

                # Plot actual classes
                sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1], hue=actual_labels, palette='viridis', ax=axes[i, j],
                                legend=False)
                axes[i, j].set_title(f"{method}\n{params['name']}")
                pbar.update(1)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def compare_projection_methods(datasets):

    quality_measures = get_available_projection_quality_measures()
    projection_methods = get_available_projection_algorithms()

    # Initialize dictionary to store cumulative metrics for averaging
    results_table = {method: {measure: 0 for measure in quality_measures}
                     for method in projection_methods}

    # Add runtime and memory usage metrics
    for method in results_table:
        results_table[method]["Runtime (s)"] = 0
        results_table[method]["Memory Usage (MiB)"] = 0

    bar_format = f"{WHITE}⌛  Testing Projection Methods...   {{l_bar}}{BLUE}{{bar}}{WHITE}{{r_bar}}{RESET}"
    with tqdm(total=len(datasets) * len(results_table.keys()), bar_format=bar_format) as pbar:
        for j, params in enumerate(datasets):
            points = load_clustering_projection_data(params['name'])

            for method, metrics in results_table.items():
                # Track memory usage and runtime for the projection method
                memory_used = memory_usage((projection_pipeline, (method, points)), max_usage=True)
                start_time = time.time()
                rd_points = projection_pipeline(method, points)
                runtime = time.time() - start_time

                # Calculate metrics
                for measure in quality_measures:
                    value = projection_quality_measure_pipeline(measure, points, rd_points)
                    if value is not None and metrics[measure] != "Error":
                        metrics[measure] += value / len(datasets)
                    else:
                        metrics[measure] = "Error"

                metrics["Runtime (s)"] += runtime / len(datasets)
                metrics["Memory Usage (MiB)"] += memory_used / len(datasets)

                pbar.update(1)

    # Print Metric Explanation
    print(f"{WHITE}")
    print("The Projection Methods were compared using the following metrics:")
    print("RMSE measures the root mean squared error between the original and reduced distances.")
    print("Kruskal Stress measures the stress of the projection.")
    print("Sammon Stress measures the stress of the projection using Sammon's mapping.")
    print("Runtime and memory usage are also tracked for comparison.")
    print(f"{RESET}", end='')

    # Print results
    for method, metrics in results_table.items():
        print(f"{WHITE}")
        print(f"Results for {method}:")
        for metric_name, value in metrics.items():
            if value == "Error":
                print(f"{metric_name}: {RED}Error (calculation not possible){WHITE}")
            elif metric_name == "Runtime (s)":
                print(f"Runtime (ms): {GREEN}{round(value * 1000, 4)}{WHITE}")
            else:
                print(f"{metric_name}: {GREEN}{round(value, 4)}{WHITE}")
        print(f"{RESET}")


if __name__ == "__main__":
    test_projection()