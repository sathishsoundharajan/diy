import random
import matplotlib.pyplot as plt

heights = [60, 61, 62, 70, 71, 72, 65, 66, 67, 75]
heights = sorted(heights)
k = 2
max_iter = 5

# Initialize centroids
centroids = random.sample(heights, k)
centroids = sorted(centroids)
print(f"Initial Centroids: {centroids}")

# Store initial state for visualization
initial_centroids = centroids.copy()
initial_clusters = None


def assign_clusters(heights, centroids):
    clusters = {c: [] for c in range(len(centroids))}
    for height in heights:
        distances = [abs(centroid - height) for centroid in centroids]
        min_distance_index = distances.index(min(distances))
        clusters[min_distance_index].append(height)
    return clusters


def take_mean_clusters(clusters, old_centroids):
    mean_centroids = []
    for clusterId in clusters:
        if clusters[clusterId]:
            mean_centroids.append(sum(clusters[clusterId]) / len(clusters[clusterId]))
        else:
            mean_centroids.append(old_centroids[clusterId])  # Keep old centroid for empty clusters
    return mean_centroids


converged = True
mean_centroids = []
mean_clusters = []

# Run K-Means
for i in range(max_iter):
    print(f"\nIteration {i + 1}")
    print(f"Centroids: {centroids}")

    clusters = assign_clusters(heights, centroids)
    print(f"Assigned Clusters: {clusters}")

    # Store initial clusters for visualization
    if i == 0:
        initial_clusters = clusters.copy()

    mean_centroids = take_mean_clusters(clusters, centroids)
    print(f"Mean Centroids: {mean_centroids}")

    mean_clusters = assign_clusters(heights, mean_centroids)
    print(f"Mean Clusters: {mean_clusters}")

    for cluster in sorted(clusters.keys()):
        if clusters[cluster] != mean_clusters[cluster]:
            converged = False
            break

    if converged:
        print(f"Converged at iteration {i + 1}")
        print(f"Final Centroids: {mean_centroids}")
        print(f"Final Clusters: {mean_clusters}")
        break

    centroids = mean_centroids

# Handle case where max iterations reached
if not converged:
    print(f"Max iterations ({max_iter}) reached without convergence")
    print(f"Final Centroids: {mean_centroids}")
    print(f"Final Clusters: {mean_clusters}")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot initial clusters
for cluster_id in initial_clusters:
    points = initial_clusters[cluster_id]
    ax1.scatter(points, [0] * len(points), label=f'Cluster {cluster_id}', s=100)
ax1.scatter(initial_centroids, [0] * len(initial_centroids), c='red', marker='x', s=200, label='Centroids')
ax1.set_title('Initial Clusters')
ax1.set_xlabel('Height')
ax1.set_yticks([])
ax1.legend()

# Plot final clusters
for cluster_id in mean_clusters:
    points = mean_clusters[cluster_id]
    ax2.scatter(points, [0] * len(points), label=f'Cluster {cluster_id}', s=100)
ax2.scatter(mean_centroids, [0] * len(mean_centroids), c='red', marker='x', s=200, label='Centroids')
ax2.set_title('Final Clusters')
ax2.set_xlabel('Height')
ax2.set_yticks([])
ax2.legend()

plt.tight_layout()
plt.show()