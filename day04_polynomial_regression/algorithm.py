import numpy as np


def polynomial_features(X, degree):
    """
    Transforms X into polynomial features up to the given degree.

    Example: X = [[2]], degree=3 -> [[2, 4, 8]]  (x, x^2, x^3)

    This is the "trick" behind polynomial regression: we don't change
    the algorithm at all, we just feed Linear Regression richer features.
    """
    X = np.array(X)
    n_samples = X.shape[0]

    # Assumes X has a single feature column (1D polynomial regression)
    x = X[:, 0]

    features = [x ** d for d in range(1, degree + 1)]
    X_poly = np.column_stack(features)

    return X_poly


class PolynomialRegressionScratch:
    """
    Polynomial Regression = Polynomial Feature Expansion + Linear Regression
    (solved via Normal Equation, same math as Day 1).

    y = b + w1*x + w2*x^2 + ... + wd*x^d
    """

    def __init__(self, degree=2):
        self.degree = degree
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X_poly = polynomial_features(X, self.degree)
        y = np.array(y).reshape(-1, 1)

        # Add bias column of 1s
        X_b = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]

        # Normal Equation: theta = (X^T X)^-1 X^T y
        theta_best = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

        self.bias = theta_best[0, 0]
        self.weights = theta_best[1:, 0]

        return self

    def predict(self, X):
        X_poly = polynomial_features(X, self.degree)
        return X_poly @ self.weights + self.bias


if __name__ == "__main__":
    # Quick manual test: y = x^2 + 2x + 1 (with noise) -> a true quadratic relationship
    np.random.seed(42)
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = (X[:, 0] ** 2) + (2 * X[:, 0]) + 1 + np.random.randn(100) * 0.5

    model = PolynomialRegressionScratch(degree=2)
    model.fit(X, y)

    print("Learned weights (should be ~[2, 1] for x, x^2):", model.weights)
    print("Learned bias (should be ~1):", model.bias)