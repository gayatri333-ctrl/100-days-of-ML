import sys
import os
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SklearnLR

from algorithm import LinearRegressionSGD

# allow importing utils/ from parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import mse, r2_score

# ---- Load Day 2's algorithm.py by exact file path (avoids name collision) ----
day02_path = os.path.join(os.path.dirname(__file__), "..", "day02_gradient_descent", "algorithm.py")
spec = importlib.util.spec_from_file_location("day02_algorithm", day02_path)
day02_algorithm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(day02_algorithm)
LinearRegressionGD = day02_algorithm.LinearRegressionGD


# ---- 1. Generate the same synthetic data as Day 1/2 ----
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 5 + 3 * X.flatten() + np.random.randn(100) * 0.5

# ---- 2. Train pure SGD (batch_size=1) ----
sgd_model = LinearRegressionSGD(learning_rate=0.05, n_epochs=50, batch_size=1)
sgd_model.fit(X, y)
sgd_preds = sgd_model.predict(X)

# ---- 3. Train Mini-batch GD (batch_size=16) ----
mb_model = LinearRegressionSGD(learning_rate=0.1, n_epochs=50, batch_size=16)
mb_model.fit(X, y)
mb_preds = mb_model.predict(X)

# ---- 4. Train Day 2's Batch GD (for comparison) ----
batch_model = LinearRegressionGD(learning_rate=0.1, n_iterations=1000)
batch_model.fit(X, y)
batch_preds = batch_model.predict(X)

# ---- 5. Train sklearn's model (ground truth reference) ----
sk_model = SklearnLR()
sk_model.fit(X, y)
sk_preds = sk_model.predict(X)

# ---- 6. Compare weights ----
print("===== Weights Comparison =====")
print(f"Pure SGD          -> weight: {sgd_model.weights[0]:.4f}, bias: {sgd_model.bias:.4f}")
print(f"Mini-batch GD     -> weight: {mb_model.weights[0]:.4f}, bias: {mb_model.bias:.4f}")
print(f"Batch GD (Day 2)  -> weight: {batch_model.weights[0]:.4f}, bias: {batch_model.bias:.4f}")
print(f"Sklearn           -> weight: {sk_model.coef_[0]:.4f}, bias: {sk_model.intercept_:.4f}")

# ---- 7. Compare metrics ----
print("\n===== Metrics Comparison =====")
print(f"Pure SGD          -> MSE: {mse(y, sgd_preds):.4f}, R2: {r2_score(y, sgd_preds):.4f}")
print(f"Mini-batch GD     -> MSE: {mse(y, mb_preds):.4f}, R2: {r2_score(y, mb_preds):.4f}")
print(f"Batch GD (Day 2)  -> MSE: {mse(y, batch_preds):.4f}, R2: {r2_score(y, batch_preds):.4f}")
print(f"Sklearn           -> MSE: {mse(y, sk_preds):.4f}, R2: {r2_score(y, sk_preds):.4f}")

# ---- 8. Plot 1: Regression lines comparison ----
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="gray", alpha=0.6, label="Data")
plt.plot(X, sgd_preds, color="orange", linewidth=2, label="Pure SGD")
plt.plot(X, mb_preds, color="green", linewidth=2, label="Mini-batch GD")
plt.plot(X, batch_preds, color="blue", linestyle="--", linewidth=2, label="Batch GD")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Day 3: SGD vs Mini-batch vs Batch Gradient Descent")
plt.legend()
plt.savefig("day03_regression_comparison.png")
plt.show()

# ---- 9. Plot 2: Loss curves side by side (shows noise difference) ----
plt.figure(figsize=(8, 5))
plt.plot(sgd_model.loss_history, color="orange", label="Pure SGD (batch_size=1)")
plt.plot(mb_model.loss_history, color="green", label="Mini-batch GD (batch_size=16)")
plt.xlabel("Epoch")
plt.ylabel("Average MSE Loss")
plt.title("Day 3: Loss Curve — SGD (noisy) vs Mini-batch (smoother)")
plt.legend()
plt.savefig("day03_loss_curve.png")
plt.show()