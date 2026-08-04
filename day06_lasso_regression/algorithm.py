import numpy as np


def soft_threshold(rho, alpha):
    """
    Soft-thresholding operator — the key math trick behind Lasso.

    It shrinks a value toward zero by `alpha`, and if the value is
    small enough, it snaps ALL THE WAY to exactly zero. This is what
    gives Lasso its feature-selection property (unlike Ridge, which
    only shrinks weights, never zeros them out).
    """
    if rho < -alpha:
        return rho + alpha
    elif rho > alpha:
        return rho - alpha
    else:
        return 0.0


class LassoRegressionScratch:
    """
    Lasso Regression = Linear Regression + L1 penalty on the weights.

    Minimizes: MSE + alpha * sum(|weights|)

    Unlike Ridge (L2), there's NO closed-form solution here because
    |weights| isn't differentiable at 0. Instead we use COORDINATE
    DESCENT: update one weight at a time, holding all others fixed,
    using the soft-thresholding formula. Repeat until convergence.

    This is the same core technique sklearn's Lasso uses internally.
    """

    def __init__(self, alpha=1.0, n_iterations=1000, tol=1e-6):
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.tol = tol
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).flatten()
        n_samples, n_features = X.shape

        # Center data so we can handle bias separately (same trick as Ridge)
        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        X_centered = X - X_mean
        y_centered = y - y_mean

        self.weights = np.zeros(n_features)

        for iteration in range(self.n_iterations):
            weights_old = self.weights.copy()

            for j in range(n_features):
                # Compute residual excluding feature j's current contribution
                residual = y_centered - X_centered @ self.weights + X_centered[:, j] * self.weights[j]

                # rho = correlation between feature j and the residual
                rho = X_centered[:, j] @ residual

                # Normalizing factor (like a per-feature "learning rate")
                z = X_centered[:, j] @ X_centered[:, j]

                # Soft-threshold update for this weight
                self.weights[j] = soft_threshold(rho, self.alpha * n_samples) / z if z != 0 else 0.0

            # Check convergence: stop early if weights barely changed
            if np.sum(np.abs(self.weights - weights_old)) < self.tol:
                break

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

    for alpha in [0, 0.01, 0.1, 1, 5]:
        model = LassoRegressionScratch(alpha=alpha)
        model.fit(X, y)
        print(f"alpha={alpha:>5} -> weight: {model.weights[0]:.4f}, bias: {model.bias:.4f}")
        # Notice: at high enough alpha, weight snaps to EXACTLY 0 (unlike Ridge)