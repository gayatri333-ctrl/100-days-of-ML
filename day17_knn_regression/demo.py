import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor as SklearnKNNRegressor

from algorithm import KNNRegressorScratch

# allow importing utils/ from parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import mse, r2_score
from utils.preprocessing import train_test_split


# ---- 1. Generate noisy sine wave data ----
np.random.seed(42)
X = np.sort(np.random.rand(100, 1) * 10, axis=0)
y = np.sin(X[:, 0]) + np.random.randn(100) * 0.15

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- 2. Compare your KNN Regressor vs sklearn's at k=5 ----
k_fixed = 5

my_model = KNNRegressorScratch(k=k_fixed)
my_model.fit(X_train, y_train)
my_preds = my_model.predict(X_test)

sk_model = SklearnKNNRegressor(n_neighbors=k_fixed)
sk_model.fit(X_train, y_train)
sk_preds = sk_model.predict(X_test)

print(f"===== Test Metrics Comparison (k={k_fixed}) =====")
print(f"My KNN      -> MSE: {mse(y_test, my_preds):.4f}, R2: {r2_score(y_test, my_preds):.4f}")
print(f"Sklearn KNN -> MSE: {mse(y_test, sk_preds):.4f}, R2: {r2_score(y_test, sk_preds):.4f}")

# ---- 3. Try different k values ----
print("\n===== Effect of k on Test MSE =====")
k_values = [1, 3, 5, 10, 20, 40]
test_mses = []

for k in k_values:
    m = KNNRegressorScratch(k=k)
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    m_score = mse(y_test, preds)
    test_mses.append(m_score)
    print(f"k={k:<4} -> Test MSE: {m_score:.4f}")

# ---- 4. Plot 1: MSE vs k ----
plt.figure(figsize=(8, 5))
plt.plot(k_values, test_mses, marker="o", color="darkorange")
plt.xlabel("k (number of neighbors)")
plt.ylabel("Test MSE")
plt.title("Day 17: KNN Regression — Effect of k on Error")
plt.savefig("day17_k_vs_mse.png")
plt.show()

# ---- 5. Plot 2: curve fit at different k values (smoothness comparison) ----
X_smooth = np.linspace(0, 10, 300).reshape(-1, 1)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
example_ks = [1, 5, 30]

for ax, k in zip(axes, example_ks):
    m = KNNRegressorScratch(k=k)
    m.fit(X_train, y_train)
    y_smooth_pred = m.predict(X_smooth)

    ax.scatter(X_train, y_train, color="gray", alpha=0.6, s=20, label="Train data")
    ax.plot(X_smooth, np.sin(X_smooth[:, 0]), color="black", linestyle=":", label="True function")
    ax.plot(X_smooth, y_smooth_pred, color="red", linewidth=2, label="KNN prediction")
    ax.set_title(f"k={k}")
    ax.set_ylim(-2, 2)
    ax.legend(fontsize=8)

plt.suptitle("Day 17: KNN Regression — Small k (jagged) vs Large k (smooth/underfit)")
plt.tight_layout()
plt.savefig("day17_fit_comparison.png")
plt.show()