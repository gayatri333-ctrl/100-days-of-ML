import sys
import os
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree

from algorithm import DecisionTreeEntropyScratch, entropy

# allow importing utils/ from parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import accuracy
from utils.preprocessing import train_test_split

# ---- Load Day 19's Gini tree by exact file path (avoids name collision) ----
day19_path = os.path.join(os.path.dirname(__file__), "..", "day19_decision_tree_gini", "algorithm.py")
spec = importlib.util.spec_from_file_location("day19_algorithm", day19_path)
day19_algorithm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(day19_algorithm)
DecisionTreeClassifierScratch = day19_algorithm.DecisionTreeClassifierScratch  # Gini version


# ---- 1. Generate data ----
np.random.seed(42)
class0 = np.random.randn(80, 2) + np.array([-1.5, -1.5])
class1 = np.random.randn(80, 2) + np.array([1.5, 1.5])
X = np.vstack([class0, class1])
y = np.array([0] * 80 + [1] * 80)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ---- 2. Compare Entropy tree vs Gini tree vs sklearn (entropy) ----
depth_fixed = 4

entropy_model = DecisionTreeEntropyScratch(max_depth=depth_fixed)
entropy_model.fit(X_train, y_train)
entropy_preds = entropy_model.predict(X_test)

gini_model = DecisionTreeClassifierScratch(max_depth=depth_fixed)
gini_model.fit(X_train, y_train)
gini_preds = gini_model.predict(X_test)

sk_model = SklearnDecisionTree(max_depth=depth_fixed, criterion="entropy", random_state=42)
sk_model.fit(X_train, y_train)
sk_preds = sk_model.predict(X_test)

print(f"===== Test Accuracy Comparison (max_depth={depth_fixed}) =====")
print(f"My Entropy Tree (Day 20) -> Accuracy: {accuracy(y_test, entropy_preds):.4f}")
print(f"My Gini Tree (Day 19)    -> Accuracy: {accuracy(y_test, gini_preds):.4f}")
print(f"Sklearn (entropy)        -> Accuracy: {accuracy(y_test, sk_preds):.4f}")

agreement = np.mean(entropy_preds == gini_preds)
print(f"\nAgreement between Entropy and Gini tree predictions: {agreement:.4f}")
print("(Gini and Entropy usually agree on MOST points, even if the trees aren't byte-identical)")

# ---- 3. Entropy sanity checks (visual proof the math is right) ----
print("\n===== Entropy Sanity Checks =====")
test_cases = {
    "Pure (all class 0)": [0, 0, 0, 0],
    "Pure (all class 1)": [1, 1, 1, 1],
    "50/50 split": [0, 1, 0, 1],
    "90/10 split": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
}
for label, y_test_case in test_cases.items():
    print(f"{label:20} -> entropy = {entropy(y_test_case):.4f}")

# ---- 4. Plot: entropy vs proportion of class 1 (the classic entropy curve) ----
proportions = np.linspace(0.001, 0.999, 200)
entropy_values = [-p * np.log2(p) - (1 - p) * np.log2(1 - p) for p in proportions]

plt.figure(figsize=(8, 5))
plt.plot(proportions, entropy_values, color="darkgreen")
plt.xlabel("Proportion of Class 1")
plt.ylabel("Entropy")
plt.title("Day 20: Binary Entropy Curve — Peaks at 50/50 Split")
plt.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="Maximum entropy (50/50)")
plt.legend()
plt.savefig("day20_entropy_curve.png")
plt.show()

# ---- 5. Plot: side-by-side decision boundaries (Gini vs Entropy) ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]

Z_entropy = entropy_model.predict(grid_points).reshape(xx.shape)
Z_gini = gini_model.predict(grid_points).reshape(xx.shape)

for ax, Z, title in zip(axes, [Z_gini, Z_entropy], ["Gini Tree (Day 19)", "Entropy Tree (Day 20)"]):
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=["#FFB6B6", "#B6D7FF"], alpha=0.6)
    ax.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color="red", edgecolor="k", s=20)
    ax.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color="blue", edgecolor="k", s=20)
    ax.set_title(title)

plt.suptitle("Day 20: Gini vs Entropy — Usually Very Similar Boundaries")
plt.tight_layout()
plt.savefig("day20_gini_vs_entropy_boundary.png")
plt.show()