import numpy as np


class LinearRegressionGD:
    """
    Linear Regression using Batch Gradient Descent.

    Instead of solving directly (Normal Equation, Day 1), this iteratively
    updates weights using the gradient of the Mean Squared Error loss:

        MSE = (1/n) * sum((y_pred - y_true)^2)

    Gradient w.r.t weights: (2/n) * X^T (X.w + b - y)
    Gradient w.r.t bias:    (2/n) * sum(X.w + b - y)
    """

    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []  # track loss per iteration for plotting

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        n_samples, n_features = X.shape

        # Initialize weights and bias to zero
        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0

        for i in range(self.n_iterations):
            y_pred = X @ self.weights + self.bias

            # Compute gradients
            error = y_pred - y
            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Track loss (MSE) for this iteration
            loss = np.mean(error ** 2)
            self.loss_history.append(loss)

        # Flatten weights for easy use later (matches Day 1's shape)
        self.weights = self.weights.flatten()

        return self

    def predict(self, X):
        X = np.array(X)
        return X @ self.weights + self.bias


if __name__ == "__main__":
    # Quick manual test: y = 3x + 5 (same data as Day 1, for comparison)
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 5 + 3 * X.flatten() + np.random.randn(100) * 0.5

    model = LinearRegressionGD(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)

    print("Learned weight (should be ~3):", model.weights)
    print("Learned bias (should be ~5):", model.bias)
    print("Final loss:", model.loss_history[-1])