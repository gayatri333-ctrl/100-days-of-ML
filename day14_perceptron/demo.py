import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron as SklearnPerceptron

from algorithm import PerceptronScratch

# allow importing utils/ from parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import accuracy


# ---- 1. Generate 2 linearly separable blobs ----
np.random.seed(42)
class0 = np.random.randn(50, 2) + np.array([-2, -2])
class1 = np.random.randn(50, 2) + np.array([2, 2])
X = np.vstack([class0, class1])
y = np.array([0] * 50 + [1] * 50)

# ---- 2. Train your from-scratch Perceptron ----
my_model = PerceptronScratch(learning_rate=0.1, n_iterations=100)
my_model.fit(X, y)
my_preds = my_model.predict(X)

# ---- 3. Train sklearn's Perceptron for comparison ----
sk_model = SklearnPerceptron(max_iter=100, eta0=0.1, random_state=42)
sk_model.fit(X, y)
sk_preds = sk_model.predict(X)

# ---- 4. Compare accuracy ----
print("===== Accuracy Comparison =====")
print(f"My Perceptron      -> Accuracy: {accuracy(y, my_preds):.4f}")
print(f"Sklearn Perceptron -> Accuracy: {accuracy(y, sk_preds):.4f}")

print(f"\nMy model converged in {len(my_model.errors_per_epoch)} epochs")
print(f"Errors per epoch: {my_model.errors_per_epoch}")

# ---- 5. Plot 1: convergence curve (errors per epoch dropping to 0) ----
plt.figure(figsize=(8, 5))
plt.plot(my_model.errors_per_epoch, marker="o", color="purple")
plt.xlabel("Epoch")
plt.ylabel("Number of Misclassifications")
plt.title("Day 14: Perceptron Convergence — Errors Drop to Zero")
plt.savefig("day14_convergence_curve.png")
plt.show()

# ---- 6. Plot 2: decision boundary ----
plt.figure(figsize=(8, 6))

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]

Z = my_model.predict(grid_points).reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=["#FFB6B6", "#B6D7FF"], alpha=0.6)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color="red", edgecolor="k", label="Class 0")
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="blue", edgecolor="k", label="Class 1")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Day 14: Perceptron Decision Boundary")
plt.legend()
plt.savefig("day14_decision_boundary.png")
plt.show()