import numpy as np


class PerceptronScratch:
    """
    The Perceptron (Rosenblatt, 1958) — the original artificial neuron
    and the ancestor of modern neural networks.

    Unlike Logistic Regression (which outputs a smooth probability via
    sigmoid), the Perceptron makes a hard binary decision directly:

        output = 1 if (X.w + b) > 0 else 0

    Learning rule (the "Perceptron Learning Rule"):
        For each misclassified point, nudge the weights toward
        fixing that specific mistake:

            w = w + learning_rate * (y_true - y_pred) * x
            b = b + learning_rate * (y_true - y_pred)

        If a point is already classified correctly, (y_true - y_pred) = 0,
        so no update happens. Only mistakes drive learning.

    IMPORTANT LIMITATION: the Perceptron is only guaranteed to converge
    if the data is LINEARLY SEPARABLE (a straight line/plane can perfectly
    split the two classes). If not, it will loop forever without settling
    — that's why we cap it with n_iterations.
    """

    def __init__(self, learning_rate=0.1, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.errors_per_epoch = []  # track how many mistakes per pass over the data

    def _step_function(self, z):
        """Hard threshold: 1 if z > 0, else 0."""
        return np.where(z > 0, 1, 0)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.n_iterations):
            errors = 0
            for idx in range(n_samples):
                x_i = X[idx]
                y_true = y[idx]

                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self._step_function(linear_output)

                update = self.learning_rate * (y_true - y_pred)

                if update != 0:
                    self.weights += update * x_i
                    self.bias += update
                    errors += 1

            self.errors_per_epoch.append(errors)

            # Early stop if a full pass had zero mistakes (converged)
            if errors == 0:
                break

        return self

    def predict(self, X):
        X = np.array(X)
        linear_output = X @ self.weights + self.bias
        return self._step_function(linear_output)


if __name__ == "__main__":
    # Quick manual test: 2 linearly separable blobs
    np.random.seed(42)
    class0 = np.random.randn(50, 2) + np.array([-2, -2])
    class1 = np.random.randn(50, 2) + np.array([2, 2])
    X = np.vstack([class0, class1])
    y = np.array([0] * 50 + [1] * 50)

    model = PerceptronScratch(learning_rate=0.1, n_iterations=100)
    model.fit(X, y)

    preds = model.predict(X)
    accuracy = np.mean(preds == y)
    print("Training accuracy:", accuracy)
    print("Epochs until convergence:", len(model.errors_per_epoch))
    print("Errors per epoch:", model.errors_per_epoch)