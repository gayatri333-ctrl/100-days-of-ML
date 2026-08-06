import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet as SklearnElasticNet

from algorithm import ElasticNetScratch

# allow importing utils/ from parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import mse, r2_score


# ---- 1. Simple 1-feature comparison (same base data as Day 1) ----
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 5 + 3 * X.flatten() + np.random.randn(100) * 0.5

alpha_fixed = 0.1
l1_ratio_fixed = 0.5

my_model = ElasticNetScratch(alpha=alpha_fixed, l1_ratio=l1_ratio_fixed)
my_model.fit(X, y)
my_preds = my_model.predict(X)

# sklearn's ElasticNet uses a slightly different alpha scaling convention,
# so we match it carefully: sklearn's objective divides the MSE term by
# n_samples already, matching our formulation when alpha is passed directly.
sk_model = SklearnElasticNet(alpha=alpha_fixed, l1_ratio=l1_ratio_fixed)
sk_model.fit(X, y)
sk_preds = sk_model.predict(X)

print(f"===== Weights Comparison (alpha={alpha_fixed}, l1_ratio={l1_ratio_fixed}) =====")
print(f"My model      -> weight: {my_model.weights[0]:.4f}, bias: {my_model.bias:.4f}")
print(f"Sklearn model -> weight: {sk_model.coef_[0]:.4f}, bias: {sk_model.intercept_:.4f}")

print("\n===== Metrics Comparison =====")
print(f"My model      -> MSE: {mse(y, my_preds):.4f}, R2: {r2_score(y, my_preds):.4f}")
print(f"Sklearn model -> MSE: {mse(y, sk_preds):.4f}, R2: {r2_score(y, sk_preds):.4f}")


# ---- 2. Spectrum demo: l1_ratio = 0 (Ridge) -> 1 (Lasso), watch weight change ----
print("\n===== l1_ratio Spectrum (alpha fixed at 0.1) =====")
print("l1_ratio=0.0 is pure Ridge, l1_ratio=1.0 is pure Lasso")
for l1_ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
    m = ElasticNetScratch(alpha=0.1, l1_ratio=l1_ratio)
    m.fit(X, y)
    print(f"l1_ratio={l1_ratio} -> weight: {m.weights[0]:.4f}")


# ---- 3. Feature selection stability demo: correlated features ----
# Two features that are almost identical (correlated) — Lasso alone tends to
# arbitrarily pick one and zero the other; Elastic Net should keep both
# with similar, shared weight due to its L2 stabilizing term.
np.random.seed(1)
n_samples = 200
base_feature = np.random.randn(n_samples)
X_corr = np.column_stack([
    base_feature,
    base_feature + np.random.randn(n_samples) * 0.01,  # nearly identical to feature 0
])
y_corr = 3 * base_feature + 2 + np.random.randn(n_samples) * 0.3

print("\n===== Correlated Features Demo (2 nearly-identical features) =====")
for l1_ratio in [1.0, 0.5, 0.0]:
    m = ElasticNetScratch(alpha=0.1, l1_ratio=l1_ratio, n_iterations=2000)
    m.fit(X_corr, y_corr)
    label = "Pure Lasso" if l1_ratio == 1.0 else "Pure Ridge" if l1_ratio == 0.0 else "Elastic Net"
    print(f"{label:15} -> weights: {np.round(m.weights, 3)}")
print("(Notice: Lasso tends to favor one feature over the other; Ridge/Elastic Net split more evenly)")

# ---- 4. Plot: fit comparison ----
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="gray", alpha=0.6, label="Data")
plt.plot(X, my_preds, color="green", linewidth=2, label="My Elastic Net")
plt.plot(X, sk_preds, color="red", linestyle="--", linewidth=2, label="Sklearn Elastic Net")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Day 7: Elastic Net Fit Comparison")
plt.legend()
plt.savefig("day07_elasticnet_fit.png")
plt.show()

# ---- 5. Plot: l1_ratio spectrum ----
l1_ratios = np.linspace(0, 1, 20)
weights_spectrum = []
for l1r in l1_ratios:
    m = ElasticNetScratch(alpha=0.1, l1_ratio=l1r)
    m.fit(X, y)
    weights_spectrum.append(m.weights[0])

plt.figure(figsize=(8, 5))
plt.plot(l1_ratios, weights_spectrum, marker="o", color="purple")
plt.xlabel("l1_ratio (0 = pure Ridge, 1 = pure Lasso)")
plt.ylabel("Learned Weight")
plt.title("Day 7: Elastic Net — Sliding Between Ridge and Lasso")
plt.savefig("day07_l1_ratio_spectrum.png")
plt.show()