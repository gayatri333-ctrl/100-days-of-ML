import sys
import os
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SklearnLR

from algorithm import LinearRegressionGD

# allow importing utils/ from parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import mse, r2_score

# ---- Load Day 1's algorithm.py by exact file path ----
# (avoids name collision since Day 2 also has a file called algorithm.py)
day01_path = os.path.join(os.path.dirname(__file__), "..", "day01_linear_regression", "algorithm.py")
spec = importlib.util.spec_from_file_location("day01_algorithm", day01_path)
day01_algorithm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(day01_algorithm)
LinearRegressionScratch = day01_algorithm.LinearRegressionScratch


# ---- 1. Generate the same synthetic data as Day 1 ----
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 5 + 3 * X.flatten() + np.random.randn(100) * 0.5

# ---- 2. Train Day 2 model: Gradient Descent ----
gd_model = LinearRegressionGD(learning_rate=0.1, n_iterations=1000)
gd_model.fit(X, y)
gd_preds = gd_model.predict(X)

# ---- 3. Train Day 1 model: Normal Equation (for comparison) ----
ne_model = LinearRegressionScratch()
ne_model.fit(X, y)
ne_preds = ne_model.predict(X)

# ---- 4. Train sklearn's model (ground truth reference) ----
sk_model = SklearnLR()
sk_model.fit(X, y)
sk_preds = sk_model.predict(X)

# ---- 5. Compare weights ----
print("===== Weights Comparison =====")
print(f"Gradient Descent  -> weight: {gd_model.weights[0]:.4f}, bias: {gd_model.bias:.4f}")
print(f"Normal Equation   -> weight: {ne_model.weights[0]:.4f}, bias: {ne_model.bias:.4f}")
print(f"Sklearn           -> weight: {sk_model.coef_[0]:.4f}, bias: {sk_model.intercept_:.4f}")

# ---- 6. Compare metrics ----
print("\n===== Metrics Comparison =====")
print(f"Gradient Descent  -> MSE: {mse(y, gd_preds):.4f}, R2: {r2_score(y, gd_preds):.4f}")
print(f"Normal Equation   -> MSE: {mse(y, ne_preds):.4f}, R2: {r2_score(y, ne_preds):.4f}")
print(f"Sklearn           -> MSE: {mse(y, sk_preds):.4f}, R2: {r2_score(y, sk_preds):.4f}")

# ---- 7. Plot 1: Regression lines comparison ----
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="gray", alpha=0.6, label="Data")
plt.plot(X, gd_preds, color="green", linewidth=2, label="Gradient Descent")
plt.plot(X, ne_preds, color="blue", linestyle="--", linewidth=2, label="Normal Equation")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Day 2: Gradient Descent vs Normal Equation")
plt.legend()
plt.savefig("day02_regression_comparison.png")
plt.show()

# ---- 8. Plot 2: Loss curve over iterations ----
plt.figure(figsize=(8, 5))
plt.plot(gd_model.loss_history, color="purple")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Day 2: Gradient Descent Loss Curve")
plt.savefig("day02_loss_curve.png")
plt.show()