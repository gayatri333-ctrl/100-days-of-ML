import numpy as np
 
 
class LinearRegressionScratch:
    """
    Linear Regression using the Normal Equation:
        theta = (X^T X)^-1 X^T y
 
    No gradient descent involved — solved directly via linear algebra.
    Good for small-to-medium datasets. (Gradient descent version comes Day 2.)
    """
 
    def __init__(self):
        self.weights = None   # coefficients (theta), includes bias term at index 0
        self.bias = None
 
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
 
        # Add a column of 1s to X for the bias/intercept term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
 
        # Normal Equation: theta = (X^T X)^-1 X^T y
        theta_best = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        # using pinv (pseudo-inverse) instead of inv -> avoids errors if X^T X is singular
 
        self.bias = theta_best[0, 0]
        self.weights = theta_best[1:, 0]
 
        return self
 
    def predict(self, X):
        X = np.array(X)
        return X @ self.weights + self.bias
 
 
if __name__ == "__main__":
    # quick manual test: y = 3x + 5 (with noise)
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 5 + 3 * X.flatten() + np.random.randn(100) * 0.5
 
    model = LinearRegressionScratch()
    model.fit(X, y)
 
    print("Learned weight (should be ~3):", model.weights)
    print("Learned bias (should be ~5):", model.bias)
 
    preds = model.predict(X[:5])
    print("Sample predictions:", preds)
    print("Actual values:", y[:5])
 