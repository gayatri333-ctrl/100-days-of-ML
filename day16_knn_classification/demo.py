import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN

from algorithm import KNNClassifierScratch

# allow importing utils/ from parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import accuracy
from utils.preprocessing import train_test_split


# ---- 1. Generate 2 overlapping-ish blobs (more realistic than fully separated) ----
np.random.seed(42)
class0 = np.random.randn(80, 2) + np.array([-1.5, -1.5])
class1 = np.random.randn(80, 2) + np.array([1.5, 1.5])
X = np.vstack([class0, class1])
y = np.array([0] * 80 + [1] * 80)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ---- 2. Compare your KNN vs sklearn's KNN at k=5 ----
k_fixed = 5

my_model = KNNClassifierScratch(k=k_fixed)
my_model.fit(X_train, y_train)
my_preds = my_model.predict(X_test)

sk_model = SklearnKNN(n_neighbors=k_fixed)
sk_model.fit(X_train, y_train)
sk_preds = sk_model.predict(X_test)

print(f"===== Test Accuracy Comparison (k={k_fixed}) =====")
print(f"My KNN      -> Accuracy: {accuracy(y_test, my_preds):.4f}")
print(f"Sklearn KNN -> Accuracy: {accuracy(y_test, sk_preds):.4f}")

# ---- 3. Try different k values, see how test accuracy changes ----
print("\n===== Effect of k on Test Accuracy =====")
k_values = [1, 3, 5, 9, 15, 25, 50]
test_accuracies = []

for k in k_values:
    m = KNNClassifierScratch(k=k)
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    acc = accuracy(y_test, preds)
    test_accuracies.append(acc)
    print(f"k={k:<4} -> Test Accuracy: {acc:.4f}")

# ---- 4. Plot 1: accuracy vs k ----
plt.figure(figsize=(8, 5))
plt.plot(k_values, test_accuracies, marker="o", color="darkorange")
plt.xlabel("k (number of neighbors)")
plt.ylabel("Test Accuracy")
plt.title("Day 16: KNN — Effect of k on Accuracy")
plt.savefig("day16_k_vs_accuracy.png")
plt.show()

# ---- 5. Plot 2: decision boundaries at 3 different k values ----
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
example_ks = [1, 5, 25]

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]

for ax, k in zip(axes, example_ks):
    m = KNNClassifierScratch(k=k)
    m.fit(X_train, y_train)
    Z = m.predict(grid_points).reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=["#FFB6B6", "#B6D7FF"], alpha=0.6)
    ax.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color="red", edgecolor="k", s=20)
    ax.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color="blue", edgecolor="k", s=20)
    ax.set_title(f"k={k}")

plt.suptitle("Day 16: KNN Decision Boundaries — Small k (jagged) vs Large k (smooth)")
plt.tight_layout()
plt.savefig("day16_decision_boundaries.png")
plt.show()