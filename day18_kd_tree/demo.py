import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree as SklearnKDTree

from algorithm import KDTree

# allow importing utils/ from parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import accuracy


# ---- 1. Correctness check against sklearn's KDTree ----
np.random.seed(42)
X = np.random.rand(500, 2) * 100
query_point = np.array([50.0, 50.0])
k = 5

my_tree = KDTree()
my_tree.fit(X)
my_distances, my_indices = my_tree.query(query_point, k=k)

sk_tree = SklearnKDTree(X)
sk_distances, sk_indices = sk_tree.query(query_point.reshape(1, -1), k=k)

print("===== Correctness Check vs Sklearn's KDTree =====")
print(f"My indices:      {sorted(my_indices)}")
print(f"Sklearn indices: {sorted(sk_indices[0].tolist())}")
print(f"Match: {sorted(my_indices) == sorted(sk_indices[0].tolist())}")


# ---- 2. Use KD-Tree to build a simple KNN classifier ----
class KNNWithKDTree:
    """A KNN classifier that uses our KD-Tree instead of brute-force search."""
    def __init__(self, k=5):
        self.k = k
        self.tree = KDTree()
        self.y_train = None

    def fit(self, X, y):
        self.tree.fit(X)
        self.y_train = np.array(y)
        return self

    def predict(self, X):
        preds = []
        for x in np.array(X):
            _, indices = self.tree.query(x, k=self.k)
            neighbor_labels = self.y_train[indices]
            values, counts = np.unique(neighbor_labels, return_counts=True)
            preds.append(values[np.argmax(counts)])
        return np.array(preds)


np.random.seed(42)
class0 = np.random.randn(60, 2) + np.array([-2, -2])
class1 = np.random.randn(60, 2) + np.array([2, 2])
X_clf = np.vstack([class0, class1])
y_clf = np.array([0] * 60 + [1] * 60)

knn_tree_model = KNNWithKDTree(k=5)
knn_tree_model.fit(X_clf, y_clf)
preds = knn_tree_model.predict(X_clf)

print(f"\n===== KD-Tree-Powered KNN Classifier =====")
print(f"Training accuracy: {accuracy(y_clf, preds):.4f}")


# ---- 3. Scaling comparison: KD-Tree vs brute-force as dataset size grows ----
print("\n===== Search Time Scaling: KD-Tree vs Brute-Force =====")
sizes = [500, 1000, 2000, 5000, 10000]
kd_times = []
brute_times = []

for size in sizes:
    X_scale = np.random.rand(size, 2) * 100
    query = np.array([50.0, 50.0])

    tree = KDTree()
    tree.fit(X_scale)
    start = time.time()
    tree.query(query, k=5)
    kd_times.append((time.time() - start) * 1000)

    start = time.time()
    distances = np.sqrt(np.sum((X_scale - query) ** 2, axis=1))
    np.argsort(distances)[:5]
    brute_times.append((time.time() - start) * 1000)

    print(f"n={size:<7} KD-Tree: {kd_times[-1]:.4f} ms | Brute-force: {brute_times[-1]:.4f} ms")

# ---- 4. Plot: search time scaling ----
plt.figure(figsize=(8, 5))
plt.plot(sizes, kd_times, marker="o", label="KD-Tree (our implementation)")
plt.plot(sizes, brute_times, marker="o", label="Brute-force (NumPy vectorized)")
plt.xlabel("Number of training points")
plt.ylabel("Search time (ms)")
plt.title("Day 18: Search Time Scaling — KD-Tree vs Brute-Force")
plt.legend()
plt.savefig("day18_scaling_comparison.png")
plt.show()