

def calculate_pagerank(graph, params):
    num_iterations = params.get('num_iterations', 100)
    damping_factor = params.get('damping_factor', 0.85)

    # Initialize PageRank values; each node starts with equal rank
    num_nodes = len(graph)
    pagerank = {node: 1 / num_nodes for node in graph}

    # Run iterative calculation
    for _ in range(num_iterations):
        new_pagerank = {}

        # Calculate PageRank for each node
        for node in graph:
            rank_sum = 0
            # Sum contributions from each incoming node
            for neighbor in graph:
                if node in graph[neighbor]:  # Check if there is an edge to `node` from `neighbor`
                    rank_sum += pagerank[neighbor] / len(graph[neighbor])

            # Apply the PageRank formula with the damping factor
            new_pagerank[node] = (1 - damping_factor) / num_nodes + damping_factor * rank_sum

        # Update PageRank values for the next iteration
        pagerank = new_pagerank

    return pagerank
