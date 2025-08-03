"""
Inverted File Index (IVF) with K-Means for Approximate Nearest-Neighbor Search

This script implements an Inverted File Index (IVF) using K-Means clustering to perform
approximate nearest-neighbor search in 2D space. It clusters a dataset of points (toys)
into k clusters using K-Means, then searches for the k nearest neighbors to a query point
by limiting the search to the nprobe closest clusters. This reduces computation compared
to brute force search over all points. Results include the closest vectors, their distances,
and their original indices in the dataset.

Key Features:
- K-Means clustering to partition data into clusters (coarse quantization).
- Euclidean distance for clustering and nearest-neighbor search.
- IVF search restricts computation to nprobe closest clusters.
- Handles empty clusters by reusing previous centroids.
- Prints detailed results with original indices for clarity.

Relevance to FAISS:
- Mimics FAISS's coarse quantizer, which clusters data to reduce search space.
- Uses Euclidean distance, a common metric in FAISS for vector similarity.
- Demonstrates approximate search, a core FAISS concept for efficiency in high-dimensional spaces.
- Prepares for advanced FAISS features like Product Quantization or HNSW indexing.

Dependencies: numpy, random
"""

import numpy as np
import random

# Sample dataset of 2D points representing toys (e.g., features like size, weight)
toys = np.array([
    [0.5, 0.8],   # Teddy Bear
    [0.6, 0.7],   # Bunny
    [2.0, 1.0],   # Toy Car
    [0.4, 0.9],   # Doll
    [0.55, 0.75]  # Robot
])

# Query point: 2D vector for which to find nearest neighbors
query = np.array([0.55, 0.75])

def assign_clusters(data_points, centroids):
    """Assign each data point to the closest centroid using Euclidean distance.

    Args:
        data_points (np.ndarray): Array of 2D points (shape: [n, 2]).
        centroids (np.ndarray): Array of current centroids (shape: [k, 2]).

    Returns:
        dict: Dictionary mapping cluster indices (int) to lists of assigned points.
    """
    clusters = {i: [] for i in range(len(centroids))}  # Initialize empty clusters
    for point in data_points:
        # Compute Euclidean distance from point to each centroid
        distances = np.sqrt(np.sum((point - centroids) ** 2, axis=1))
        # Assign point to cluster with minimum distance
        min_distance_index = np.argmin(distances)
        clusters[min_distance_index].append(point.tolist())  # Convert to list for consistency
    return clusters

def take_mean_centroids(clusters, centroids):
    """Recalculate centroids as the mean of their assigned points.

    Args:
        clusters (dict): Dictionary mapping cluster indices to lists of points.
        centroids (np.ndarray): Current centroids (shape: [k, 2]) for empty clusters.

    Returns:
        np.ndarray: Array of new centroids (means of non-empty clusters or old centroids).
    """
    mean_centroids = []
    for cluster_idx in sorted(clusters.keys()):
        if clusters[cluster_idx]:
            # Compute mean of x and y coordinates for non-empty cluster
            mean_centroids.append(np.mean(clusters[cluster_idx], axis=0))
        else:
            # Reuse old centroid for empty clusters to avoid errors
            mean_centroids.append(centroids[cluster_idx])
    return np.array(mean_centroids)

def kmeans(data_points, k):
    """Perform K-Means clustering to partition data into k clusters.

    Args:
        data_points (np.ndarray): Array of 2D points to cluster (shape: [n, 2]).
        k (int): Number of clusters.

    Returns:
        tuple: (final_centroids, final_clusters)
            - final_centroids (np.ndarray): Final centroids (shape: [k, 2]).
            - final_clusters (dict): Cluster assignments (index to list of points).
    """
    random.seed(42)  # Set seed for reproducible centroid initialization
    max_iter = 10  # Maximum iterations to prevent infinite loops

    # Step 1: Initialize centroids by randomly selecting k points
    centroid_indices = random.sample(range(len(data_points)), k)
    centroids = data_points[centroid_indices].copy()
    print("Initial Centroids:", centroids)

    final_centroids = []
    final_clusters = []

    # Step 2: Main K-Means loop
    for iter in range(max_iter):
        print(f"K-Means Iteration: {iter + 1}")

        # Assign points to nearest centroids
        clusters = assign_clusters(data_points, centroids)

        # Recalculate centroids as means of assigned points
        mean_centroids = take_mean_centroids(clusters, centroids)

        # Check for convergence: stop if centroids stabilize
        if np.allclose(centroids, mean_centroids, rtol=1e-5, atol=1e-8):
            print("K-Means Converged.")
            final_centroids = mean_centroids
            final_clusters = clusters
            break

        centroids = mean_centroids

        # Handle non-convergence
        if iter == max_iter - 1:
            print("Max iterations reached without convergence.")
            final_centroids = mean_centroids
            final_clusters = clusters

    return final_centroids, final_clusters

class IVF:
    """Inverted File Index (IVF) for fast approximate nearest-neighbor search.

    Clusters data using K-Means and searches only within the closest clusters
    to reduce computation compared to brute force search.
    """

    def __init__(self, data, nlist):
        """Initialize the IVF index by clustering the data with K-Means.

        Args:
            data (np.ndarray): Array of 2D points to index (shape: [n, 2]).
            nlist (int): Number of clusters (centroids) for K-Means.
        """
        self.data = data  # Store dataset
        self.nlist = nlist  # Number of clusters
        print("\nBuilding IVF Index with K-Means...")
        # Run K-Means to get centroids and clusters
        self.centroids, self.clusters = kmeans(data, nlist)
        print(f"IVF Index built successfully.\n Centroids: {self.centroids} \n Clusters: {self.clusters}")

    def search(self, query, k, nprobe=1):
        """Search for the k nearest neighbors to a query vector using IVF.

        Args:
            query (np.ndarray): 2D query point (shape: [2,]).
            k (int): Number of nearest neighbors to retrieve.
            nprobe (int): Number of closest clusters to search.

        Returns:
            tuple: (closest_vectors, distances)
                - closest_vectors (np.ndarray): k closest points (shape: [k, 2]).
                - distances (np.ndarray): Euclidean distances to the k closest points.
        """
        # Step 1: Find the nprobe closest centroids to the query
        distance_to_centroids = np.sqrt(np.sum((self.centroids - query) ** 2, axis=1))
        closest_centroid_indices = np.argsort(distance_to_centroids)[:nprobe]

        # Step 2: Collect points from the nprobe closest clusters
        search_vectors_list = []
        for index in closest_centroid_indices:
            search_vectors_list.extend(self.clusters[index])

        # Handle case where no points are in the selected clusters
        if not search_vectors_list:
            return np.array([]), np.array([])

        # Convert list of points to NumPy array
        search_vectors = np.array(search_vectors_list)

        # Step 3: Compute Euclidean distances from query to points in selected clusters
        distances = np.sqrt(np.sum((search_vectors - query) ** 2, axis=1))

        # Step 4: Find the k closest points within the selected clusters
        closest_indices = np.argsort(distances)[:k]

        # Return the k closest points and their distances
        return search_vectors[closest_indices], distances[closest_indices]

# Main execution
ivf = IVF(toys, nlist=2)  # Build IVF index with 2 clusters
# Search for 3 nearest neighbors to the query point
closest_vectors, distances = ivf.search(query, k=3, nprobe=1)

# Print search results
print("IVF Search Results:")
print("Closest toys (vectors):", closest_vectors)
print("Their distances:", distances)

# Map returned vectors to their original indices in the toys array
for i, vector in enumerate(closest_vectors):
    # Find index where toys array matches the returned vector
    original_idx = np.where(np.all(toys == vector, axis=1))[0][0]
    print(f"Original Toy Index: {original_idx}, Vector: {toys[original_idx]}, Distance: {distances[i]:.3f}")