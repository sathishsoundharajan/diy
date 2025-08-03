"""
Brute Force Nearest-Neighbor Search

This script implements a brute force nearest-neighbor search to find the k closest
points (toys) to a query point in 2D space using Euclidean distance. It computes
distances from the query to all points, sorts them, and returns the indices and
distances of the k nearest points. The results are printed with toy vectors and
their distances for clarity.

Key Features:
- Uses NumPy for efficient vector operations.
- Computes Euclidean distance for all points in a single pass.
- Returns the k nearest neighbors based on sorted distances.
- Simple and interpretable, ideal for understanding similarity search.

Relevance to FAISS:
- Demonstrates brute force nearest-neighbor search, which FAISS optimizes using
  indexing structures (e.g., k-d trees, HNSW) and quantization (e.g., Product Quantization).
- Uses Euclidean distance, a common metric in FAISS for vector similarity.
- Serves as a baseline for understanding FAISS's efficiency improvements in
  high-dimensional vector search.

Dependencies: numpy
"""

import numpy as np

# Input data: 2D points representing toys with features (e.g., size, weight)
toys = np.array([
    [0.5, 0.8],   # Teddy Bear
    [0.6, 0.7],   # Bunny
    [2.0, 1.0],   # Toy Car
    [0.4, 0.9],   # Doll
    [0.55, 0.75]  # Robot
])

# Query point: 2D vector for which to find similar toys
query = np.array([0.55, 0.75])

def find_similar_toys(toys, query, k=3):
    """Find the k nearest toys to a query point using Euclidean distance.

    Args:
        toys (np.ndarray): Array of 2D points representing toys (shape: [n, 2]).
        query (np.ndarray): 2D query point (shape: [2,]).
        k (int, optional): Number of nearest neighbors to return. Defaults to 3.

    Returns:
        tuple: (closest_indices, closest_distances)
            - closest_indices (np.ndarray): Indices of the k closest toys.
            - closest_distances (np.ndarray): Euclidean distances to the k closest toys.
    """
    # Step 1: Compute Euclidean distances from query to all toys
    differences = toys - query  # Vectorized subtraction: toys - query
    squared_differences = differences ** 2  # Square each dimension difference
    sum_squared = squared_differences.sum(axis=1)  # Sum squared differences across dimensions
    distances = np.sqrt(sum_squared)  # Take square root to get Euclidean distance

    # Step 2: Find the k closest toys by sorting distances
    closest_indices = np.argsort(distances)[:k]  # Indices of k smallest distances
    closest_distances = distances[closest_indices]  # Corresponding distances

    return closest_indices, closest_distances

# Run the search
indices, distances = find_similar_toys(toys, query, k=3)

# Print results
print("Closest toys (indices):", indices)
print("Their distances:", distances)
for i, idx in enumerate(indices):
    print(f"Toy {idx} with vector {toys[idx]} has distance {distances[i]:.3f}")