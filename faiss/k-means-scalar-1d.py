"""
1D K-Means Clustering Implementation

This script implements the K-Means clustering algorithm for 1D data points (e.g., heights).
It groups points into k clusters by iteratively assigning points to the nearest centroid
and updating centroids as the mean of assigned points. Convergence is detected when
cluster assignments stabilize. The script visualizes initial and final clusters using
Matplotlib scatter plots.

Key Features:
- Random centroid initialization from data points.
- Absolute distance for cluster assignment.
- Handles empty clusters by reusing previous centroids.
- Order-independent cluster comparison for convergence.
- Visualizes initial and final clusters with centroids marked.

Relevance to FAISS:
- Demonstrates basic clustering, similar to FAISS's coarse quantizer for vector search.
- Uses distance-based assignment, a core concept in FAISS's nearest-neighbor search.
- Prepares for scaling to higher dimensions, as in FAISS's high-dimensional vector handling.

Dependencies: random, matplotlib
"""

import random
import matplotlib.pyplot as plt

# Input data: list of 1D points (heights in inches)
heights = [60, 61, 62, 70, 71, 72, 65, 66, 67, 75]
heights = sorted(heights)  # Sort for consistent processing
k = 2  # Number of clusters
max_iter = 5  # Maximum iterations to prevent infinite loops

# Initialize centroids by randomly selecting k points from the dataset
centroids = random.sample(heights, k)
centroids = sorted(centroids)  # Sort for consistent initial display
print(f"Initial Centroids: {centroids}")

# Store initial state for visualization
initial_centroids = centroids.copy()  # Copy to preserve initial centroids
initial_clusters = None  # Will store clusters from first iteration

def assign_clusters(heights, centroids):
    """Assign each point to the nearest centroid based on absolute distance.

    Args:
        heights (list): List of 1D points (floats or integers).
        centroids (list): List of current centroids (floats or integers).

    Returns:
        dict: Dictionary mapping cluster indices (int) to lists of assigned points.
    """
    clusters = {c: [] for c in range(len(centroids))}  # Initialize empty clusters
    for height in heights:
        # Compute absolute distance from point to each centroid
        distances = [abs(centroid - height) for centroid in centroids]
        # Assign point to cluster with minimum distance
        min_distance_index = distances.index(min(distances))
        clusters[min_distance_index].append(height)
    return clusters

def take_mean_clusters(clusters, old_centroids):
    """Compute the mean of each cluster to update centroids.

    Args:
        clusters (dict): Dictionary mapping cluster indices to lists of points.
        old_centroids (list): List of previous centroids for empty clusters.

    Returns:
        list: List of new centroids (means of non-empty clusters or old centroids).
    """
    mean_centroids = []
    for clusterId in sorted(clusters.keys()):
        if clusters[clusterId]:
            # Compute mean of points in non-empty cluster
            mean_centroids.append(sum(clusters[clusterId]) / len(clusters[clusterId]))
        else:
            # Reuse old centroid for empty clusters to avoid errors
            mean_centroids.append(old_centroids[clusterId])
    return mean_centroids

def clusters_equal(c1, c2):
    """Compare two cluster assignments for equality.

    Args:
        c1 (dict): First cluster assignment (index to list of points).
        c2 (dict): Second cluster assignment (index to list of points).

    Returns:
        bool: True if clusters have identical points (order-independent).
    """
    for key in sorted(c1.keys()):
        # Handle empty clusters
        if not c1[key] and not c2[key]:
            continue  # Empty clusters are equal
        if len(c1[key]) != len(c2[key]):
            return False  # Different sizes mean not equal
        # Sort points for order-independent comparison
        if sorted(c1[key]) != sorted(c2[key]):
            return False
    return True

# Run K-Means algorithm
for i in range(max_iter):
    print(f"\nIteration {i + 1}")
    print(f"Centroids: {centroids}")

    # Assign points to nearest centroids
    clusters = assign_clusters(heights, centroids)
    print(f"Assigned Clusters: {clusters}")

    # Store initial clusters for visualization
    if i == 0:
        initial_clusters = clusters.copy()

    # Compute new centroids as means of clusters
    mean_centroids = take_mean_clusters(clusters, centroids)
    print(f"Mean Centroids: {mean_centroids}")

    # Reassign points to new centroids
    mean_clusters = assign_clusters(heights, mean_centroids)
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

# Plot initial clusters
for cluster_id in initial_clusters:
    points = initial_clusters[cluster_id]
    if points:  # Skip empty clusters
        ax1.scatter(points, [0] * len(points), label=f'Cluster {cluster_id}', s=100, alpha=0.6)
ax1.scatter(initial_centroids, [0] * len(initial_centroids), c='red', marker='x', s=200,
            label='Centroids', linewidths=3)
ax1.set_title('Initial Clusters')
ax1.set_xlabel('Height')
ax1.set_yticks([])  # No y-axis ticks for 1D data
ax1.legend()

# Plot final clusters
for cluster_id in mean_clusters:
    points = mean_clusters[cluster_id]
    if points:  # Skip empty clusters
        ax2.scatter(points, [0] * len(points), label=f'Cluster {cluster_id}', s=100, alpha=0.6)
ax2.scatter(mean_centroids, [0] * len(mean_centroids), c='red', marker='x', s=200,
            label='Centroids', linewidths=3)
ax2.set_title('Final Clusters')
ax2.set_xlabel('Height')
ax2.set_yticks([])
ax2.legend()

plt.tight_layout()  # Adjust subplot spacing
plt.show()