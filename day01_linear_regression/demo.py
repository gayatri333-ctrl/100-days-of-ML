
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SklearnLR
 
from algorithm import LinearRegressionScratch
import sys
import os
 
# allow importing utils/ from parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import mse, r2_score
 
 
# ---- 1. Generate synthetic data ----
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 5 + 3 * X.flatten() + np.random.randn(100) * 0.5
 
# ---- 2. Train your from-scratch model ----
my_model = LinearRegressionScratch()
my_model.fit(X, y)
my_preds = my_model.predict(X)
 
# ---- 3. Train sklearn's model for comparison ----
sk_model = SklearnLR()
sk_model.fit(X, y)
sk_preds = sk_model.predict(X)
 
# ---- 4. Compare results ----
print("===== Weights Comparison =====")
print(f"My model      -> weight: {my_model.weights[0]:.4f}, bias: {my_model.bias:.4f}")
print(f"Sklearn model -> weight: {sk_model.coef_[0]:.4f}, bias: {sk_model.intercept_:.4f}")
 
print("\n===== Metrics Comparison =====")
print(f"My model      -> MSE: {mse(y, my_preds):.4f}, R2: {r2_score(y, my_preds):.4f}")
print(f"Sklearn model -> MSE: {mse(y, sk_preds):.4f}, R2: {r2_score(y, sk_preds):.4f}")
 
# ---- 5. Visualize ----
plt.scatter(X, y, color="gray", alpha=0.6, label="Data")
plt.plot(X, my_preds, color="blue", linewidth=2, label="My Model")
plt.plot(X, sk_preds, color="red", linestyle="--", linewidth=2, label="Sklearn Model")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Day 1: Linear Regression (Normal Equation) — Scratch vs Sklearn")
plt.legend()
plt.savefig("day01_result.png")
plt.show()
 