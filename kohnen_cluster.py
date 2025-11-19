import numpy as np

np.set_printoptions(precision=3, suppress=True)

# --------------------------------------------
# Compute Euclidean distance
# --------------------------------------------
def compute_distances(x, weights):
    # weights: (4 features × 2 neurons)
    # x: (4,)
    distances = np.sum((weights.T - x)**2, axis=1)
    return distances

# --------------------------------------------
# Winner neuron
# --------------------------------------------
def select_winner(distances):
    return np.argmin(distances)

# --------------------------------------------
# Weight update rule (move towards input)
# --------------------------------------------
def update_weights(weights, x, winner, alpha):
    weights[:, winner] = weights[:, winner] + alpha * (x - weights[:, winner])
    return weights

# --------------------------------------------
# Train for 1 epoch
# --------------------------------------------
def train_one_epoch(vectors, weights, alpha):
    cluster_assignments = []
    for x in vectors:
        distances = compute_distances(x, weights)
        winner = select_winner(distances)
        weights = update_weights(weights, x, winner, alpha)
        cluster_assignments.append(winner)
    return weights, cluster_assignments

# --------------------------------------------
# MAIN
# --------------------------------------------
if __name__ == "__main__":

    print("Enter 5 vectors (each of length 4): ")
    n = 5
    m = 2  # 2 clusters

    vectors = np.zeros((n, 4))

    for i in range(n):
        vec = input(f"Enter Vector {i+1}: ")
        vectors[i] = np.array(list(map(float, vec.split())))

    # Initial weights (as per your PDF)
    weights = np.array([
        [0.2, 0.9],
        [0.4, 0.7],
        [0.6, 0.5],
        [0.8, 0.3]
    ])

    print("\nInitial Weights:\n", weights)

    alpha = 0.5
    epochs = 100

    for epoch in range(1, epochs + 1):
        weights, cluster_assignments = train_one_epoch(vectors, weights, alpha)
        alpha = alpha * 0.5  # learning rate decay

    print(f"\nFinal Weights after {epochs} epochs:\n", np.round(weights, 3))

    print("\nFinal Cluster Assignments:")
    for i, cluster in enumerate(cluster_assignments, start=1):
        print(f"Vector {i} → Cluster {cluster + 1}")
