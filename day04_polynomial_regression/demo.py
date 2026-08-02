import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.preprocessing import PolynomialFeatures

from algorithm import PolynomialRegressionScratch

# allow importing utils/ from parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import mse, r2_score


# ---- 1. Generate non-linear synthetic data: y = x^2 + 2x + 1 ----
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = (X[:, 0] ** 2) + (2 * X[:, 0]) + 1 + np.random.randn(100) * 0.5

# ---- 2. Train your from-scratch Polynomial Regression (degree=2) ----
my_model = PolynomialRegressionScratch(degree=2)
my_model.fit(X, y)
my_preds = my_model.predict(X)

# ---- 3. Train sklearn's PolynomialFeatures + LinearRegression pipeline ----
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_sklearn = poly.fit_transform(X)
sk_model = SklearnLR()
sk_model.fit(X_poly_sklearn, y)
sk_preds = sk_model.predict(X_poly_sklearn)

# ---- 4. Also fit a plain (degree=1) Linear Regression for contrast ----
linear_model = SklearnLR()
linear_model.fit(X, y)
linear_preds = linear_model.predict(X)

# ---- 5. Compare weights ----
print("===== Weights Comparison (degree=2) =====")
print(f"My model      -> weights (x, x^2): {my_model.weights}, bias: {my_model.bias:.4f}")
print(f"Sklearn model -> weights (x, x^2): {sk_model.coef_}, bias: {sk_model.intercept_:.4f}")

# ---- 6. Compare metrics ----
print("\n===== Metrics Comparison =====")
print(f"My model (degree=2)    -> MSE: {mse(y, my_preds):.4f}, R2: {r2_score(y, my_preds):.4f}")
print(f"Sklearn (degree=2)     -> MSE: {mse(y, sk_preds):.4f}, R2: {r2_score(y, sk_preds):.4f}")
print(f"Plain Linear (degree=1)-> MSE: {mse(y, linear_preds):.4f}, R2: {r2_score(y, linear_preds):.4f}")
print("(Notice how much worse plain linear regression fits this curved data)")

# ---- 7. Plot: curve fit comparison ----
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="gray", alpha=0.5, label="Data (true curve: y = x^2 + 2x + 1)")

# sort X for a smooth line plot
sort_idx = np.argsort(X[:, 0])
plt.plot(X[sort_idx], my_preds[sort_idx], color="green", linewidth=2, label="My Polynomial Model (degree=2)")
plt.plot(X[sort_idx], sk_preds[sort_idx], color="red", linestyle="--", linewidth=2, label="Sklearn Polynomial Model")
plt.plot(X[sort_idx], linear_preds[sort_idx], color="blue", linestyle=":", linewidth=2, label="Plain Linear Regression (underfits)")

plt.xlabel("X")
plt.ylabel("y")
plt.title("Day 4: Polynomial Regression — Fitting a Curve")
plt.legend()
plt.savefig("day04_polynomial_fit.png")
plt.show()