import numpy as np


def soft_threshold(rho, alpha):
    """Same soft-thresholding operator from Day 6 (Lasso)."""
    if rho < -alpha:
        return rho + alpha
    elif rho > alpha:
        return rho - alpha
    else:
        return 0.0


class ElasticNetScratch:
    """
    Elastic Net = Linear Regression + BOTH L1 and L2 penalties.

    Minimizes:
        MSE + alpha * [ l1_ratio * sum(|w|) + (1 - l1_ratio) * 0.5 * sum(w^2) ]

    l1_ratio controls the mix:
        l1_ratio = 1.0  -> pure Lasso (L1 only)
        l1_ratio = 0.0  -> pure Ridge (L2 only)
        0 < l1_ratio < 1 -> blend of both (this is the "Elastic Net" sweet spot)

    Why use it? Lasso alone can behave erratically when features are
    correlated (it arbitrarily picks one and zeroes the other). Elastic
    Net's added L2 term stabilizes that, while still keeping some of
    Lasso's feature-selection ability.

    Solved via coordinate descent, same technique as Day 6, just with
    an extra L2 term added to the per-feature update rule.
    """

    def __init__(self, alpha=1.0, l1_ratio=0.5, n_iterations=1000, tol=1e-6):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_iterations = n_iterations
        self.tol = tol
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).flatten()
        n_samples, n_features = X.shape

        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        X_centered = X - X_mean
        y_centered = y - y_mean

        self.weights = np.zeros(n_features)

        # Split total alpha into its L1 and L2 components based on l1_ratio
        l1_penalty = self.alpha * self.l1_ratio
        l2_penalty = self.alpha * (1 - self.l1_ratio)

        for iteration in range(self.n_iterations):
            weights_old = self.weights.copy()

            for j in range(n_features):
                residual = y_centered - X_centered @ self.weights + X_centered[:, j] * self.weights[j]
                rho = X_centered[:, j] @ residual
                z = X_centered[:, j] @ X_centered[:, j]

                # Elastic Net coordinate update:
                # numerator gets soft-thresholded (L1 effect),
                # denominator gets an extra term (L2 effect, shrinks smoothly)
                numerator = soft_threshold(rho, l1_penalty * n_samples)
                denominator = z + l2_penalty * n_samples

                self.weights[j] = numerator / denominator if denominator != 0 else 0.0

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

    for l1_ratio in [0.0, 0.5, 1.0]:
        model = ElasticNetScratch(alpha=0.1, l1_ratio=l1_ratio)
        model.fit(X, y)
        label = "Pure Ridge" if l1_ratio == 0.0 else "Pure Lasso" if l1_ratio == 1.0 else "Elastic Net (mixed)"
        print(f"l1_ratio={l1_ratio} ({label}) -> weight: {model.weights[0]:.4f}, bias: {model.bias:.4f}")