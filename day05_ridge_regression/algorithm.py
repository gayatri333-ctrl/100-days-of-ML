import numpy as np


class RidgeRegressionScratch:
    """
    Ridge Regression = Linear Regression + L2 penalty on the weights.

    Instead of just minimizing MSE, we minimize:
        MSE + alpha * sum(weights^2)

    This shrinks large weights toward zero, which helps prevent overfitting
    (especially useful when features are correlated or when there are many
    features relative to samples).

    Closed-form solution (modified Normal Equation):
        theta = (X^T X + alpha * I)^-1 X^T y

    Note: the bias term is NOT regularized (standard convention), so we
    handle it separately from the weights.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha  # regularization strength (lambda)
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        n_samples, n_features = X.shape

        # Center X and y so we can solve for weights without regularizing the bias.
        # This is the standard trick: regularizing the bias term is undesirable
        # since it doesn't relate to "complexity" the way feature weights do.
        X_mean = X.mean(axis=0)
        y_mean = y.mean()

        X_centered = X - X_mean
        y_centered = y - y_mean

        # Ridge closed-form: theta = (X^T X + alpha * I)^-1 X^T y
        identity = np.eye(n_features)
        theta = np.linalg.pinv(X_centered.T @ X_centered + self.alpha * identity) @ X_centered.T @ y_centered

        self.weights = theta.flatten()
        self.bias = y_mean - X_mean @ self.weights

        return self

    def predict(self, X):
        X = np.array(X)
        return X @ self.weights + self.bias


if __name__ == "__main__":
    # Quick manual test: y = 3x + 5 (same base data as Day 1)
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 5 + 3 * X.flatten() + np.random.randn(100) * 0.5

    for alpha in [0, 1, 10, 100]:
        model = RidgeRegressionScratch(alpha=alpha)
        model.fit(X, y)
        print(f"alpha={alpha:>4} -> weight: {model.weights[0]:.4f}, bias: {model.bias:.4f}")
        # As alpha increases, weight should shrink toward 0 (visible regularization effect)