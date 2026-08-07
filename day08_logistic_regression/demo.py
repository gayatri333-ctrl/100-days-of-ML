import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as SklearnLogReg

from algorithm import LogisticRegressionScratch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import accuracy, precision, recall, f1_score, confusion_matrix


# ---- 1. Generate 2 well-separated blobs (binary classification data) ----
np.random.seed(42)
class0 = np.random.randn(50, 2) + np.array([-2, -2])
class1 = np.random.randn(50, 2) + np.array([2, 2])
X = np.vstack([class0, class1])
y = np.array([0] * 50 + [1] * 50)

# ---- 2. Train your from-scratch model ----
my_model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
my_model.fit(X, y)
my_preds = my_model.predict(X)

# ---- 3. Train sklearn's model for comparison ----
sk_model = SklearnLogReg()
sk_model.fit(X, y)
sk_preds = sk_model.predict(X)

# ---- 4. Compare metrics using YOUR metrics.py from Day 1's utils ----
print("===== My Model Metrics =====")
print(f"Accuracy:  {accuracy(y, my_preds):.4f}")
print(f"Precision: {precision(y, my_preds):.4f}")
print(f"Recall:    {recall(y, my_preds):.4f}")
print(f"F1 Score:  {f1_score(y, my_preds):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y, my_preds)}")

print("\n===== Sklearn Model Metrics =====")
print(f"Accuracy:  {accuracy(y, sk_preds):.4f}")
print(f"Precision: {precision(y, sk_preds):.4f}")
print(f"Recall:    {recall(y, sk_preds):.4f}")
print(f"F1 Score:  {f1_score(y, sk_preds):.4f}")

# ---- 5. Plot: loss curve over training ----
plt.figure(figsize=(8, 5))
plt.plot(my_model.loss_history, color="purple")
plt.xlabel("Iteration")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Day 8: Logistic Regression Loss Curve")
plt.savefig("day08_loss_curve.png")
plt.show()

# ---- 6. Plot: decision boundary ----
plt.figure(figsize=(8, 6))

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]

Z = my_model.predict_proba(grid_points).reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.6)
plt.colorbar(label="P(class = 1)")
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color="blue", edgecolor="k", label="Class 0")
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="red", edgecolor="k", label="Class 1")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Day 8: Logistic Regression Decision Boundary")
plt.legend()
plt.savefig("day08_decision_boundary.png")
plt.show()