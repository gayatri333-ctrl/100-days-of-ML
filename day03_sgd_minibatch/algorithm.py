import numpy as np


class LinearRegressionSGD:
    """
    Linear Regression using Stochastic Gradient Descent (SGD) and
    Mini-batch Gradient Descent.

    Unlike Batch GD (Day 2), which uses the ENTIRE dataset to compute
    the gradient at every step, this updates weights using only a small
    subset of samples at a time:

        batch_size = 1            -> pure Stochastic Gradient Descent
        1 < batch_size < n_samples -> Mini-batch Gradient Descent
        batch_size = n_samples     -> equivalent to Batch GD (Day 2)

    This makes each update noisier but much faster per step, and often
    converges faster in practice on large datasets.
    """

    def __init__(self, learning_rate=0.1, n_epochs=50, batch_size=1, random_state=42):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.loss_history = []  # average loss per epoch

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        n_samples, n_features = X.shape

        rng = np.random.default_rng(self.random_state)

        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0

        for epoch in range(self.n_epochs):
            # Shuffle data at the start of every epoch
            indices = rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_losses = []

            # Walk through the data in batches of batch_size
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred = X_batch @ self.weights + self.bias
                error = y_pred - y_batch

                # Gradients computed only on this batch
                dw = (2 / X_batch.shape[0]) * (X_batch.T @ error)
                db = (2 / X_batch.shape[0]) * np.sum(error)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

                epoch_losses.append(np.mean(error ** 2))

            # Track average loss across all batches this epoch
            self.loss_history.append(np.mean(epoch_losses))

        self.weights = self.weights.flatten()
        return self

    def predict(self, X):
        X = np.array(X)
        return X @ self.weights + self.bias


if __name__ == "__main__":
    # Quick manual test: y = 3x + 5 (same data as Day 1/2, for comparison)
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 5 + 3 * X.flatten() + np.random.randn(100) * 0.5

    print("--- Pure SGD (batch_size=1) ---")
    sgd_model = LinearRegressionSGD(learning_rate=0.05, n_epochs=50, batch_size=1)
    sgd_model.fit(X, y)
    print("weight:", sgd_model.weights, "bias:", sgd_model.bias)

    print("\n--- Mini-batch GD (batch_size=16) ---")
    mb_model = LinearRegressionSGD(learning_rate=0.1, n_epochs=50, batch_size=16)
    mb_model.fit(X, y)
    print("weight:", mb_model.weights, "bias:", mb_model.bias)