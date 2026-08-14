import sys
import os
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

# allow importing utils/ from parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import mse, rmse, mae, r2_score
from utils.preprocessing import train_test_split, StandardScaler, cross_val_score


def load_module(day_folder, file_name="algorithm.py"):
    """Helper to import algorithm.py from any day's folder by exact path,
    avoiding the naming collision problem we hit back on Day 2."""
    path = os.path.join(os.path.dirname(__file__), "..", day_folder, file_name)
    spec = importlib.util.spec_from_file_location(day_folder, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- Load models from previous days ----
LinearRegressionScratch = load_module("day01_linear_regression").LinearRegressionScratch
RidgeRegressionScratch = load_module("day05_ridge_regression").RidgeRegressionScratch
LassoRegressionScratch = load_module("day06_lasso_regression").LassoRegressionScratch


# =====================================================================
# STEP 1: Generate a synthetic house price dataset
# =====================================================================
# Features: square_footage, num_bedrooms, num_bathrooms, house_age, distance_to_city
# Price depends realistically on these, with some noise and one irrelevant-ish feature.
np.random.seed(42)
n_samples = 300

square_footage = np.random.normal(1800, 600, n_samples).clip(500, 5000)
num_bedrooms = np.random.randint(1, 6, n_samples)
num_bathrooms = np.random.randint(1, 4, n_samples)
house_age = np.random.randint(0, 80, n_samples)
distance_to_city = np.random.normal(15, 8, n_samples).clip(0.5, 50)

# True underlying price formula (what we're trying to recover)
price = (
    150 * square_footage
    + 8000 * num_bedrooms
    + 12000 * num_bathrooms
    - 500 * house_age
    - 1200 * distance_to_city
    + 20000
    + np.random.normal(0, 15000, n_samples)  # noise
)

X = np.column_stack([square_footage, num_bedrooms, num_bathrooms, house_age, distance_to_city])
y = price
feature_names = ["square_footage", "num_bedrooms", "num_bathrooms", "house_age", "distance_to_city"]

print("===== Dataset =====")
print(f"Samples: {n_samples}, Features: {X.shape[1]}")
print(f"Price range: ${y.min():,.0f} - ${y.max():,.0f}")


# =====================================================================
# STEP 2: Train/test split (Day 11)
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain samples: {len(X_train)}, Test samples: {len(X_test)}")


# =====================================================================
# STEP 3: Scale features (Day 12) — fit on train, apply to both
# =====================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =====================================================================
# STEP 4: Cross-validate multiple models (Day 11's cross_val_score) on TRAIN data
# =====================================================================
print("\n===== 5-Fold Cross-Validation on Training Data (R2 score) =====")

candidates = {
    "Linear Regression": (LinearRegressionScratch, {}),
    "Ridge (alpha=1)": (RidgeRegressionScratch, {"alpha": 1.0}),
    "Ridge (alpha=10)": (RidgeRegressionScratch, {"alpha": 10.0}),
    "Lasso (alpha=0.1)": (LassoRegressionScratch, {"alpha": 0.1}),
    "Lasso (alpha=1.0)": (LassoRegressionScratch, {"alpha": 1.0}),
}

cv_results = {}
for name, (model_class, kwargs) in candidates.items():
    scores = cross_val_score(model_class, X_train_scaled, y_train, n_splits=5, metric_fn=r2_score, **kwargs)
    cv_results[name] = scores
    print(f"{name:20} -> mean R2: {scores.mean():.4f}  (folds: {np.round(scores, 3)})")

best_name = max(cv_results, key=lambda k: cv_results[k].mean())
print(f"\nBest model by cross-validation: {best_name}")


# =====================================================================
# STEP 5: Train the best model on FULL training data, evaluate on TEST set
# =====================================================================
best_model_class, best_kwargs = candidates[best_name]
final_model = best_model_class(**best_kwargs)
final_model.fit(X_train_scaled, y_train)
test_preds = final_model.predict(X_test_scaled)

print(f"\n===== Final Evaluation on Held-Out Test Set ({best_name}) =====")
print(f"MSE:  {mse(y_test, test_preds):,.2f}")
print(f"RMSE: {rmse(y_test, test_preds):,.2f}")
print(f"MAE:  {mae(y_test, test_preds):,.2f}")
print(f"R2:   {r2_score(y_test, test_preds):.4f}")

print(f"\nLearned weights ({feature_names}):")
print(np.round(final_model.weights, 2))
print(f"Bias: {final_model.bias:.2f}")


# =====================================================================
# STEP 6: Visualizations
# =====================================================================

# Plot 1: Cross-validation comparison across models
plt.figure(figsize=(9, 5))
names = list(cv_results.keys())
means = [cv_results[n].mean() for n in names]
stds = [cv_results[n].std() for n in names]
plt.bar(names, means, yerr=stds, capsize=5, color="steelblue")
plt.ylabel("Mean R2 Score (5-Fold CV)")
plt.title("Day 15: Model Comparison via Cross-Validation")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("day15_model_comparison.png")
plt.show()

# Plot 2: Predicted vs Actual prices on test set
plt.figure(figsize=(7, 7))
plt.scatter(y_test, test_preds, alpha=0.6, color="green")
min_val, max_val = min(y_test.min(), test_preds.min()), max(y_test.max(), test_preds.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Perfect prediction")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Day 15: Predicted vs Actual House Prices ({best_name})")
plt.legend()
plt.savefig("day15_predicted_vs_actual.png")
plt.show()