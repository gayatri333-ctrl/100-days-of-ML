import sys
import os
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

# allow importing utils/ from parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import mse
from utils.preprocessing import train_test_split

# ---- Load Day 4's PolynomialRegressionScratch by exact file path ----
day04_path = os.path.join(os.path.dirname(__file__), "..", "day04_polynomial_regression", "algorithm.py")
spec = importlib.util.spec_from_file_location("day04_algorithm", day04_path)
day04_algorithm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(day04_algorithm)
PolynomialRegressionScratch = day04_algorithm.PolynomialRegressionScratch


# ---- 1. Generate noisy non-linear data (true function is a sine wave) ----
np.random.seed(42)
n_samples = 60
X = np.sort(np.random.rand(n_samples, 1) * 6 - 3, axis=0)  # range roughly [-3, 3]
true_function = np.sin(X[:, 0])
y = true_function + np.random.randn(n_samples) * 0.3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---- 2. Train polynomial models of increasing degree, track train/test error ----
degrees = list(range(1, 16))
train_errors = []
test_errors = []
models = {}

for degree in degrees:
    model = PolynomialRegressionScratch(degree=degree)
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_errors.append(mse(y_train, train_preds))
    test_errors.append(mse(y_test, test_preds))
    models[degree] = model

print("===== Bias-Variance Tradeoff: Train vs Test Error by Model Complexity =====")
print(f"{'Degree':<8}{'Train MSE':<15}{'Test MSE':<15}Diagnosis")
for degree, tr_err, te_err in zip(degrees, train_errors, test_errors):
    if degree <= 2:
        diagnosis = "Underfitting (high bias)"
    elif te_err > tr_err * 3:
        diagnosis = "Overfitting (high variance)"
    else:
        diagnosis = "Reasonable fit"
    print(f"{degree:<8}{tr_err:<15.4f}{te_err:<15.4f}{diagnosis}")

best_degree = degrees[np.argmin(test_errors)]
print(f"\nBest degree by test error: {best_degree} (test MSE = {min(test_errors):.4f})")

# ---- 3. Plot 1: Train vs Test error curve (the classic U-shape for test error) ----
plt.figure(figsize=(8, 5))
plt.plot(degrees, train_errors, marker="o", color="blue", label="Train Error")
plt.plot(degrees, test_errors, marker="o", color="red", label="Test Error")
plt.axvline(best_degree, color="green", linestyle="--", alpha=0.6, label=f"Best degree ({best_degree})")
plt.xlabel("Polynomial Degree (Model Complexity)")
plt.ylabel("Mean Squared Error")
plt.title("Day 13: Bias-Variance Tradeoff — Train vs Test Error")
plt.legend()
plt.savefig("day13_bias_variance_curve.png")
plt.show()

# ---- 4. Plot 2: Visualize 3 specific fits - underfit, good fit, overfit ----
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
example_degrees = [1, best_degree, 15]
titles = ["Underfitting (degree=1)", f"Good Fit (degree={best_degree})", "Overfitting (degree=15)"]

X_smooth = np.linspace(-3, 3, 300).reshape(-1, 1)

for ax, degree, title in zip(axes, example_degrees, titles):
    model = models[degree]
    y_smooth_pred = model.predict(X_smooth)

    ax.scatter(X_train, y_train, color="gray", alpha=0.6, s=20, label="Train data")
    ax.plot(X_smooth, np.sin(X_smooth[:, 0]), color="black", linestyle=":", label="True function")
    ax.plot(X_smooth, y_smooth_pred, color="red", linewidth=2, label="Model prediction")
    ax.set_title(title)
    ax.set_ylim(-2, 2)
    ax.legend(fontsize=8)

plt.suptitle("Day 13: Underfitting vs Good Fit vs Overfitting")
plt.tight_layout()
plt.savefig("day13_fit_examples.png")
plt.show()