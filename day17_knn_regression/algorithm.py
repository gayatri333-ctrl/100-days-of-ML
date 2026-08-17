import numpy as np


def euclidean_distance(a, b):
    """Straight-line distance between two points."""
    return np.sqrt(np.sum((a - b) ** 2))


class KNNRegressorScratch:
    """
    K-Nearest Neighbors Regressor.

    Nearly identical to Day 16's KNN Classifier — same "lazy learning"
    idea (fit just memorizes the data). The only real difference is the
    final aggregation step at prediction time:

        Classification (Day 16): majority VOTE among k neighbors' labels
        Regression (today):      AVERAGE of k neighbors' target values

    Prediction for a new point x:
        1. Find the k closest training points to x
        2. Average their y-values -> that's the prediction

    Same k tradeoff as before:
        - Small k: prediction follows local data closely (can be noisy/jagged)
        - Large k: prediction is smoother (can underfit if too large,
          since it averages over points that are less locally relevant)
    """

    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def _predict_single(self, x):
        distances = np.array([euclidean_distance(x, x_train) for x_train in self.X_train])
        k_nearest_indices = np.argsort(distances)[: self.k]
        k_nearest_values = self.y_train[k_nearest_indices]

        # Simple average of the k nearest neighbors' target values
        return np.mean(k_nearest_values)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])


if __name__ == "__main__":
    # Quick manual test: y = sin(x), a smooth non-linear curve
    np.random.seed(42)
    X = np.sort(np.random.rand(80, 1) * 10, axis=0)
    y = np.sin(X[:, 0]) + np.random.randn(80) * 0.1

    model = KNNRegressorScratch(k=5)
    model.fit(X, y)

    preds = model.predict(X)
    mse = np.mean((y - preds) ** 2)
    print("Training MSE (k=5):", mse)