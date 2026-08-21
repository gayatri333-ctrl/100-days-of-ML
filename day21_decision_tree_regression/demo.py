import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor

from algorithm import DecisionTreeRegressorScratch

# allow importing utils/ from parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import mse, r2_score
from utils.preprocessing import train_test_split


# ---- 1. Generate noisy sine wave data (same as Day 17, for comparison) ----
np.random.seed(42)
X = np.sort(np.random.rand(100, 1) * 10, axis=0)
y = np.sin(X[:, 0]) + np.random.randn(100) * 0.15

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- 2. Compare your Decision Tree Regressor vs sklearn's at max_depth=4 ----
depth_fixed = 4

my_model = DecisionTreeRegressorScratch(max_depth=depth_fixed)
my_model.fit(X_train, y_train)
my_preds = my_model.predict(X_test)

sk_model = SklearnDecisionTreeRegressor(max_depth=depth_fixed, random_state=42)
sk_model.fit(X_train, y_train)
sk_preds = sk_model.predict(X_test)

print(f"===== Test Metrics Comparison (max_depth={depth_fixed}) =====")
print(f"My Tree      -> MSE: {mse(y_test, my_preds):.4f}, R2: {r2_score(y_test, my_preds):.4f}")
print(f"Sklearn Tree -> MSE: {mse(y_test, sk_preds):.4f}, R2: {r2_score(y_test, sk_preds):.4f}")

# ---- 3. Effect of max_depth ----
print("\n===== Effect of max_depth on Test MSE =====")
depths = [1, 2, 3, 5, 8, 15]
test_mses = []

for depth in depths:
    m = DecisionTreeRegressorScratch(max_depth=depth)
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    m_score = mse(y_test, preds)
    test_mses.append(m_score)
    print(f"max_depth={depth:<4} -> Test MSE: {m_score:.4f}")

# ---- 4. Plot 1: MSE vs max_depth ----
plt.figure(figsize=(8, 5))
plt.plot(depths, test_mses, marker="o", color="darkorange")
plt.xlabel("max_depth")
plt.ylabel("Test MSE")
plt.title("Day 21: Decision Tree Regression — Effect of max_depth")
plt.savefig("day21_depth_vs_mse.png")
plt.show()

# ---- 5. Plot 2: staircase-shaped fit at different depths ----
X_smooth = np.linspace(0, 10, 300).reshape(-1, 1)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
example_depths = [1, 4, 15]

for ax, depth in zip(axes, example_depths):
    m = DecisionTreeRegressorScratch(max_depth=depth)
    m.fit(X_train, y_train)
    y_smooth_pred = m.predict(X_smooth)

    ax.scatter(X_train, y_train, color="gray", alpha=0.6, s=20, label="Train data")
    ax.plot(X_smooth, np.sin(X_smooth[:, 0]), color="black", linestyle=":", label="True function")
    ax.plot(X_smooth, y_smooth_pred, color="red", linewidth=2, label="Tree prediction")
    ax.set_title(f"max_depth={depth}")
    ax.set_ylim(-2, 2)
    ax.legend(fontsize=8)

plt.suptitle("Day 21: Decision Tree Regression — Notice the STAIRCASE shape (unlike KNN's smooth curve)")
plt.tight_layout()
plt.savefig("day21_staircase_fit.png")
plt.show()