import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso as SklearnLasso

from algorithm import LassoRegressionScratch

# allow importing utils/ from parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import mse, r2_score


# ---- 1. Simple 1-feature comparison (same base data as Day 1) ----
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 5 + 3 * X.flatten() + np.random.randn(100) * 0.5

alpha_fixed = 0.1

my_model = LassoRegressionScratch(alpha=alpha_fixed)
my_model.fit(X, y)
my_preds = my_model.predict(X)

sk_model = SklearnLasso(alpha=alpha_fixed)
sk_model.fit(X, y)
sk_preds = sk_model.predict(X)

print(f"===== Weights Comparison (alpha={alpha_fixed}) =====")
print(f"My model      -> weight: {my_model.weights[0]:.4f}, bias: {my_model.bias:.4f}")
print(f"Sklearn model -> weight: {sk_model.coef_[0]:.4f}, bias: {sk_model.intercept_:.4f}")

print("\n===== Metrics Comparison =====")
print(f"My model      -> MSE: {mse(y, my_preds):.4f}, R2: {r2_score(y, my_preds):.4f}")
print(f"Sklearn model -> MSE: {mse(y, sk_preds):.4f}, R2: {r2_score(y, sk_preds):.4f}")


# ---- 2. Feature selection demo: multiple features, some USELESS ----
# X has 5 features, but y only truly depends on features 0 and 2.
# A good Lasso model should push weights for features 1, 3, 4 toward 0.
np.random.seed(0)
n_samples = 200
X_multi = np.random.randn(n_samples, 5)
true_weights = np.array([4.0, 0.0, -2.5, 0.0, 0.0])  # only features 0 and 2 matter
y_multi = X_multi @ true_weights + 3 + np.random.randn(n_samples) * 0.5

print("\n===== Feature Selection Demo (5 features, only 2 truly matter) =====")
print(f"True weights: {true_weights}")

for alpha in [0.001, 0.05, 0.2, 0.5, 1.0]:
    m = LassoRegressionScratch(alpha=alpha, n_iterations=2000)
    m.fit(X_multi, y_multi)
    print(f"alpha={alpha:>6} -> learned weights: {np.round(m.weights, 3)}")

# ---- 3. Plot 1: fit comparison on simple data ----
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="gray", alpha=0.6, label="Data")
plt.plot(X, my_preds, color="green", linewidth=2, label=f"My Lasso (alpha={alpha_fixed})")
plt.plot(X, sk_preds, color="red", linestyle="--", linewidth=2, label=f"Sklearn Lasso (alpha={alpha_fixed})")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Day 6: Lasso Regression Fit Comparison")
plt.legend()
plt.savefig("day06_lasso_fit.png")
plt.show()

# ---- 4. Plot 2: how each weight shrinks to zero as alpha increases ----
alphas_path = np.linspace(0.001, 1.0, 30)
weights_path = []

for a in alphas_path:
    m = LassoRegressionScratch(alpha=a, n_iterations=2000)
    m.fit(X_multi, y_multi)
    weights_path.append(m.weights)

weights_path = np.array(weights_path)

plt.figure(figsize=(8, 5))
for feature_idx in range(5):
    plt.plot(alphas_path, weights_path[:, feature_idx], label=f"Feature {feature_idx} (true={true_weights[feature_idx]})")
plt.xlabel("Alpha")
plt.ylabel("Learned Weight")
plt.title("Day 6: Lasso Feature Selection — Useless Features Drop to Exactly 0")
plt.legend()
plt.savefig("day06_feature_selection_path.png")
plt.show()