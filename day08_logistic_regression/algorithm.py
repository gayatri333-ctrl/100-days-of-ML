import numpy as np


def sigmoid(z):
    """
    Sigmoid function: squashes any real number into range (0, 1),
    which we interpret as a probability.

        sigmoid(z) = 1 / (1 + e^-z)

    Clipped to avoid overflow warnings for very large/small z.
    """
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


class LogisticRegressionScratch:
    """
    Logistic Regression for BINARY classification (0 or 1).

    Unlike Linear Regression, we don't predict y directly — we predict
    the PROBABILITY that y=1, using:

        p = sigmoid(X.w + b)

    Trained by minimizing Binary Cross-Entropy loss (not MSE):

        loss = -(1/n) * sum( y*log(p) + (1-y)*log(1-p) )

    Optimized via gradient descent, same update pattern as Day 2,
    but the gradient formula changes because the loss function changed.
    Conveniently, the gradient works out to almost the exact same form
    as linear regression's gradient:

        dw = (1/n) * X^T (p - y)
        db = (1/n) * sum(p - y)
    """

    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        n_samples, n_features = X.shape

        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0

        for i in range(self.n_iterations):
            z = X @ self.weights + self.bias
            p = sigmoid(z)

            # Gradients of Binary Cross-Entropy loss
            error = p - y
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Track loss (with small epsilon to avoid log(0))
            eps = 1e-15
            p_clipped = np.clip(p, eps, 1 - eps)
            loss = -np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
            self.loss_history.append(loss)

        self.weights = self.weights.flatten()
        return self

    def predict_proba(self, X):
        """Returns the raw probability of class 1."""
        X = np.array(X)
        return sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        """Returns the hard class label (0 or 1) using a probability threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)


if __name__ == "__main__":
    # Quick manual test: 2 well-separated blobs of points, classify which blob
    np.random.seed(42)
    class0 = np.random.randn(50, 2) + np.array([-2, -2])
    class1 = np.random.randn(50, 2) + np.array([2, 2])
    X = np.vstack([class0, class1])
    y = np.array([0] * 50 + [1] * 50)

    model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)

    preds = model.predict(X)
    accuracy = np.mean(preds == y)
    print("Training accuracy:", accuracy)
    print("Final loss:", model.loss_history[-1])