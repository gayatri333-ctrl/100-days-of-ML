import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge as SklearnRidge

from algorithm import RidgeRegressionScratch

# allow importing utils/ from parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import mse, r2_score


# ---- 1. Generate synthetic data (same base as Day 1): y = 3x + 5 ----
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 5 + 3 * X.flatten() + np.random.randn(100) * 0.5

# ---- 2. Compare your Ridge vs sklearn's Ridge at a fixed alpha ----
alpha_fixed = 10.0

my_model = RidgeRegressionScratch(alpha=alpha_fixed)
my_model.fit(X, y)
my_preds = my_model.predict(X)

sk_model = SklearnRidge(alpha=alpha_fixed)
sk_model.fit(X, y)
sk_preds = sk_model.predict(X)

print(f"===== Weights Comparison (alpha={alpha_fixed}) =====")
print(f"My model      -> weight: {my_model.weights[0]:.4f}, bias: {my_model.bias:.4f}")
print(f"Sklearn model -> weight: {sk_model.coef_[0]:.4f}, bias: {sk_model.intercept_:.4f}")

print("\n===== Metrics Comparison =====")
print(f"My model      -> MSE: {mse(y, my_preds):.4f}, R2: {r2_score(y, my_preds):.4f}")
print(f"Sklearn model -> MSE: {mse(y, sk_preds):.4f}, R2: {r2_score(y, sk_preds):.4f}")

# ---- 3. Regularization path: how weight shrinks as alpha increases ----
alphas = [0, 0.1, 1, 10, 50, 100, 500, 1000]
my_weights_path = []
sk_weights_path = []

for a in alphas:
    m = RidgeRegressionScratch(alpha=a)
    m.fit(X, y)
    my_weights_path.append(m.weights[0])

    s = SklearnRidge(alpha=a)
    s.fit(X, y)
    sk_weights_path.append(s.coef_[0])

print("\n===== Regularization Path (weight shrinkage) =====")
for a, w_mine, w_sk in zip(alphas, my_weights_path, sk_weights_path):
    print(f"alpha={a:>6} -> my weight: {w_mine:.4f} | sklearn weight: {w_sk:.4f}")

# ---- 4. Plot: regression fit comparison ----
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="gray", alpha=0.6, label="Data")
plt.plot(X, my_preds, color="green", linewidth=2, label=f"My Ridge (alpha={alpha_fixed})")
plt.plot(X, sk_preds, color="red", linestyle="--", linewidth=2, label=f"Sklearn Ridge (alpha={alpha_fixed})")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Day 5: Ridge Regression Fit Comparison")
plt.legend()
plt.savefig("day05_ridge_fit.png")
plt.show()

# ---- 5. Plot: regularization path (weight shrinking as alpha grows) ----
plt.figure(figsize=(8, 5))
plt.plot(alphas, my_weights_path, marker="o", color="green", label="My Ridge")
plt.plot(alphas, sk_weights_path, marker="x", color="red", linestyle="--", label="Sklearn Ridge")
plt.xscale("log")
plt.xlabel("Alpha (log scale)")
plt.ylabel("Learned Weight")
plt.title("Day 5: Regularization Path — Weight Shrinks as Alpha Increases")
plt.legend()
plt.savefig("day05_regularization_path.png")
plt.show()