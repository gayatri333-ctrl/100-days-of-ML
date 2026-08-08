import numpy as np


def softmax(z):
    """
    Softmax function: generalizes sigmoid to MULTIPLE classes.

    Converts a vector of raw scores into a probability distribution
    that sums to 1 across all classes:

        softmax(z)_i = e^(z_i) / sum(e^(z_j) for all j)

    We subtract the max value per row before exponentiating purely
    for numerical stability (prevents overflow) — this doesn't change
    the result mathematically, since softmax is shift-invariant.
    """
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot_encode(y, n_classes):
    """Converts integer labels [0, 2, 1, ...] into one-hot rows."""
    y = np.array(y).astype(int)
    one_hot = np.zeros((y.shape[0], n_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


class SoftmaxRegressionScratch:
    """
    Logistic Regression generalized to MULTICLASS classification.

    Instead of one weight vector (binary case), we now have one weight
    vector PER CLASS. Each class gets its own "score", and softmax turns
    those scores into a probability distribution over all classes.

        scores = X.W + b       (W has shape [n_features, n_classes])
        probs  = softmax(scores)

    Trained by minimizing Categorical Cross-Entropy loss:

        loss = -(1/n) * sum( y_true_one_hot * log(probs) )

    The gradient has the exact same elegant form as binary logistic
    regression, just now in matrix form across all classes at once:

        dW = (1/n) * X^T (probs - y_one_hot)
        db = (1/n) * sum(probs - y_one_hot)
    """

    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None   # shape: [n_features, n_classes]
        self.bias = None      # shape: [n_classes]
        self.n_classes = None
        self.loss_history = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.n_classes = len(np.unique(y))

        y_one_hot = one_hot_encode(y, self.n_classes)

        self.weights = np.zeros((n_features, self.n_classes))
        self.bias = np.zeros(self.n_classes)

        for i in range(self.n_iterations):
            scores = X @ self.weights + self.bias
            probs = softmax(scores)

            error = probs - y_one_hot
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error, axis=0)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            eps = 1e-15
            probs_clipped = np.clip(probs, eps, 1 - eps)
            loss = -np.mean(np.sum(y_one_hot * np.log(probs_clipped), axis=1))
            self.loss_history.append(loss)

        return self

    def predict_proba(self, X):
        X = np.array(X)
        scores = X @ self.weights + self.bias
        return softmax(scores)

    def predict(self, X):
        """Returns the class with the highest predicted probability."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


if __name__ == "__main__":
    # Quick manual test: 3 well-separated blobs, classify into 3 classes
    np.random.seed(42)
    class0 = np.random.randn(40, 2) + np.array([-3, -3])
    class1 = np.random.randn(40, 2) + np.array([3, -3])
    class2 = np.random.randn(40, 2) + np.array([0, 3])
    X = np.vstack([class0, class1, class2])
    y = np.array([0] * 40 + [1] * 40 + [2] * 40)

    model = SoftmaxRegressionScratch(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)

    preds = model.predict(X)
    accuracy = np.mean(preds == y)
    print("Training accuracy:", accuracy)
    print("Final loss:", model.loss_history[-1])