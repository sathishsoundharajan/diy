"""
2D K-Means Clustering Implementation

This script implements the K-Means clustering algorithm for 2D data points.
It groups points into k clusters by iteratively assigning points to the nearest
centroid (using Euclidean distance) and updating centroids as the mean of assigned
points. Convergence is detected when cluster assignments stabilize. The script
visualizes initial and final clusters using Matplotlib scatter plots.

Key Features:
- Random centroid initialization from data points.
- Euclidean distance for cluster assignment.
- Handles empty clusters by reusing previous centroids.
- Order-independent cluster comparison with floating-point tolerance.
- Visualizes initial and final clusters with centroids marked.

Relevance to FAISS:
- Uses vector operations (Euclidean distance, mean computation) similar to FAISS's vector search.
- Demonstrates clustering, a core concept in FAISS's coarse quantizer for efficient similarity search.
- Prepares for high-dimensional clustering and indexing, as used in FAISS.

Dependencies: random, numpy, matplotlib
"""

import random
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
random.seed(42)

# Input data: 2D points as NumPy array
data_points = np.array([
    [1, 1], [1.5, 2], [2, 1], [2.5, 2.5], [0.5, 0.5],  # Cluster 1-like
    [8, 8], [8.5, 9], [7.5, 8.5], [9, 8], [9.5, 9.5],  # Cluster 2-like
    [3, 7], [3.5, 8], [4, 7.5], [4.5, 7], [5, 8.5]     # Cluster 3-like
])

k = 3  # Number of clusters
max_iter = 100  # Maximum iterations to prevent infinite loops

# Initialize centroids by randomly selecting k points from the dataset
centroid_indices = random.sample(range(len(data_points)), k)
print(f"Initial Centroid Indices: {centroid_indices}")
centroids = data_points[centroid_indices].copy()
print(f"Initial Centroids: {centroids}")

# Store initial state for visualization
initial_centroids = centroids.copy()  # Copy to preserve initial centroids
initial_clusters = None  # Will store clusters from first iteration

def assign_clusters(data, centroids):
    """Assign each point to the nearest centroid based on Euclidean distance.

    Args:
        data (np.ndarray): Array of 2D points (shape: [n, 2]).
        centroids (np.ndarray): Array of current centroids (shape: [k, 2]).

    Returns:
        dict: Dictionary mapping cluster indices (int) to lists of assigned points.
    """
    clusters = {i: [] for i in range(len(centroids))}  # Initialize empty clusters
    for point in data:
        # Compute Euclidean distance from point to each centroid
        distances = [np.sqrt(np.sum((point - centroid)**2)) for centroid in centroids]
        # Assign point to cluster with minimum distance
        min_distance_index = np.argmin(distances)
        clusters[min_distance_index].append(point.tolist())  # Convert to list for consistency
    return clusters

def take_mean_centroid(clusters, old_centroids):
    """Compute the mean of each cluster to update centroids.

    Args:
        clusters (dict): Dictionary mapping cluster indices to lists of points.
        old_centroids (np.ndarray): Array of previous centroids (shape: [k, 2]).

    Returns:
        np.ndarray: Array of new centroids (means of non-empty clusters or old centroids).
    """
    mean_centroids = []
    for cluster in sorted(clusters.keys()):
        if clusters[cluster]:
            # Compute mean of x and y coordinates for non-empty cluster
            mean_centroids.append(np.mean(clusters[cluster], axis=0))
        else:
            # Reuse old centroid for empty clusters to avoid errors
            mean_centroids.append(old_centroids[cluster])
    return np.array(mean_centroids)

def clusters_equal(c1, c2):
    """Compare two cluster assignments for equality.

    Args:
        c1 (dict): First cluster assignment (index to list of points).
        c2 (dict): Second cluster assignment (index to list of points).

    Returns:
        bool: True if clusters have identical points (order-independent, with tolerance).
    """
    for key in sorted(c1.keys()):
        # Handle empty clusters
        if not c1[key] and not c2[key]:
            continue  # Empty clusters are equal
        if len(c1[key]) != len(c2[key]):
            return False  # Different sizes mean not equal
        # Convert lists to sorted tuples for order-independent comparison
        a = sorted(tuple(p) for p in c1[key])
        b = sorted(tuple(p) for p in c2[key])
        # Use np.allclose for floating-point tolerance
        if not np.allclose(a, b, rtol=1e-5, atol=1e-8):
            return False
    return True

# Run K-Means algorithm
for i in range(max_iter):
    print(f"\nIteration {i + 1}")
    clusters = assign_clusters(data_points, centroids)
    print(f"Clusters: {clusters}")

    # Store initial clusters for visualization
    if i == 0:
        initial_clusters = clusters.copy()

    # Compute new centroids as means of clusters
    mean_centroids = take_mean_centroid(clusters, centroids)
    print(f"Mean Centroids: {mean_centroids}")

    # Reassign points to new centroids
    mean_clusters = assign_clusters(data_points, mean_centroids)
    print(f"Mean Clusters: {mean_clusters}")

    # Check for convergence (same cluster assignments)
    if clusters_equal(clusters, mean_clusters):
        print(f"Converged at iteration {i + 1}")
        print(f"Final Centroids: {mean_centroids}")
        print(f"Final Clusters: {mean_clusters}")
        break

    # Update centroids for next iteration
    centroids = mean_centroids

    # Handle case where max iterations reached without convergence
    if i == max_iter - 1:
        print(f"Max iterations ({max_iter}) reached without convergence")
        print(f"Final Centroids: {mean_centroids}")
        print(f"Final Clusters: {mean_clusters}")

# Visualize initial and final clusters
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # Create two subplots
colors = ['blue', 'orange', 'green']  # Colors for clusters

# Plot initial clusters
for cluster_id in initial_clusters:
    if initial_clusters[cluster_id]:
        x = [p[0] for p in initial_clusters[cluster_id]]
        y = [p[1] for p in initial_clusters[cluster_id]]
        ax1.scatter(x, y, c=colors[cluster_id], label=f'Cluster {cluster_id}', s=100, alpha=0.6)
ax1.scatter([c[0] for c in initial_centroids], [c[1] for c in initial_centroids],
            c='red', marker='x', s=200, label='Centroids', linewidths=3)
ax1.set_title('Initial Clusters')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.legend()

# Plot final clusters
for cluster_id in mean_clusters:
    if mean_clusters[cluster_id]:
        x = [p[0] for p in mean_clusters[cluster_id]]
        y = [p[1] for p in mean_clusters[cluster_id]]
        ax2.scatter(x, y, c=colors[cluster_id], label=f'Cluster {cluster_id}', s=100, alpha=0.6)
ax2.scatter([c[0] for c in mean_centroids], [c[1] for c in mean_centroids],
            c='red', marker='x', s=200, label='Centroids', linewidths=3)
ax2.set_title('Final Clusters')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.legend()

plt.tight_layout()  # Adjust subplot spacing
plt.show()