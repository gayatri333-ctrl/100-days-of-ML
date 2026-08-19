import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree

from algorithm import DecisionTreeClassifierScratch

# allow importing utils/ from parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import accuracy
from utils.preprocessing import train_test_split


# ---- 1. Generate data with a bit of overlap (more realistic, shows overfitting) ----
np.random.seed(42)
class0 = np.random.randn(80, 2) + np.array([-1.5, -1.5])
class1 = np.random.randn(80, 2) + np.array([1.5, 1.5])
X = np.vstack([class0, class1])
y = np.array([0] * 80 + [1] * 80)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ---- 2. Compare your Decision Tree vs sklearn's at max_depth=3 ----
depth_fixed = 3

my_model = DecisionTreeClassifierScratch(max_depth=depth_fixed)
my_model.fit(X_train, y_train)
my_preds = my_model.predict(X_test)

sk_model = SklearnDecisionTree(max_depth=depth_fixed, criterion="gini", random_state=42)
sk_model.fit(X_train, y_train)
sk_preds = sk_model.predict(X_test)

print(f"===== Test Accuracy Comparison (max_depth={depth_fixed}) =====")
print(f"My Tree      -> Accuracy: {accuracy(y_test, my_preds):.4f}")
print(f"Sklearn Tree -> Accuracy: {accuracy(y_test, sk_preds):.4f}")

# ---- 3. Effect of max_depth: watch overfitting emerge ----
print("\n===== Effect of max_depth on Train vs Test Accuracy =====")
depths = [1, 2, 3, 5, 8, 12, 20]
train_accs, test_accs = [], []

for depth in depths:
    m = DecisionTreeClassifierScratch(max_depth=depth)
    m.fit(X_train, y_train)
    train_acc = accuracy(y_train, m.predict(X_train))
    test_acc = accuracy(y_test, m.predict(X_test))
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    print(f"max_depth={depth:<4} -> Train: {train_acc:.4f} | Test: {test_acc:.4f}")

# ---- 4. Plot 1: train vs test accuracy as depth increases ----
plt.figure(figsize=(8, 5))
plt.plot(depths, train_accs, marker="o", color="blue", label="Train Accuracy")
plt.plot(depths, test_accs, marker="o", color="red", label="Test Accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Day 19: Decision Tree — Overfitting as Depth Increases")
plt.legend()
plt.savefig("day19_depth_vs_accuracy.png")
plt.show()

# ---- 5. Plot 2: decision boundary comparison (shallow vs deep tree) ----
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
example_depths = [1, 3, 20]

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]

for ax, depth in zip(axes, example_depths):
    m = DecisionTreeClassifierScratch(max_depth=depth)
    m.fit(X_train, y_train)
    Z = m.predict(grid_points).reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=["#FFB6B6", "#B6D7FF"], alpha=0.6)
    ax.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color="red", edgecolor="k", s=20)
    ax.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color="blue", edgecolor="k", s=20)
    ax.set_title(f"max_depth={depth}")

plt.suptitle("Day 19: Decision Tree Boundaries — Notice the Blocky/Rectangular Shape")
plt.tight_layout()
plt.savefig("day19_decision_boundaries.png")
plt.show()