import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as SklearnLogReg

from algorithm import SoftmaxRegressionScratch

# allow importing utils/ from parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import accuracy


# ---- 1. Generate 3 well-separated blobs (multiclass data) ----
np.random.seed(42)
class0 = np.random.randn(40, 2) + np.array([-3, -3])
class1 = np.random.randn(40, 2) + np.array([3, -3])
class2 = np.random.randn(40, 2) + np.array([0, 3])
X = np.vstack([class0, class1, class2])
y = np.array([0] * 40 + [1] * 40 + [2] * 40)

# ---- 2. Train your from-scratch model ----
my_model = SoftmaxRegressionScratch(learning_rate=0.1, n_iterations=1000)
my_model.fit(X, y)
my_preds = my_model.predict(X)

# ---- 3. Train sklearn's model for comparison ----
# sklearn's LogisticRegression auto-detects multiclass and uses softmax internally
sk_model = SklearnLogReg(max_iter=1000)
sk_model.fit(X, y)
sk_preds = sk_model.predict(X)

# ---- 4. Compare accuracy ----
print("===== Accuracy Comparison =====")
print(f"My model      -> Accuracy: {accuracy(y, my_preds):.4f}")
print(f"Sklearn model -> Accuracy: {accuracy(y, sk_preds):.4f}")

# ---- 5. Show a few sample predictions with their probabilities ----
print("\n===== Sample Predictions (first 5 points) =====")
probs = my_model.predict_proba(X[:5])
for i in range(5):
    print(f"Point {i}: true={y[i]}, predicted={my_preds[i]}, probabilities={np.round(probs[i], 3)}")

# ---- 6. Plot: loss curve over training ----
plt.figure(figsize=(8, 5))
plt.plot(my_model.loss_history, color="purple")
plt.xlabel("Iteration")
plt.ylabel("Categorical Cross-Entropy Loss")
plt.title("Day 9: Softmax Regression Loss Curve")
plt.savefig("day09_loss_curve.png")
plt.show()

# ---- 7. Plot: multiclass decision boundary (3 colored regions) ----
plt.figure(figsize=(8, 6))

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]

Z = my_model.predict(grid_points).reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5], colors=["#FFB6B6", "#B6D7FF", "#B6FFC1"], alpha=0.6)
colors = ["red", "blue", "green"]
for class_label in range(3):
    plt.scatter(X[y == class_label][:, 0], X[y == class_label][:, 1],
                color=colors[class_label], edgecolor="k", label=f"Class {class_label}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Day 9: Softmax Regression — 3-Class Decision Boundary")
plt.legend()
plt.savefig("day09_decision_boundary.png")
plt.show()