import numpy as np
from collections import Counter


def euclidean_distance(a, b):
    """Straight-line distance between two points."""
    return np.sqrt(np.sum((a - b) ** 2))


class KNNClassifierScratch:
    """
    K-Nearest Neighbors Classifier.

    Unlike every model so far, KNN has NO real "training" step — it's
    called a "lazy learner" because fit() just stores the data. All the
    actual work happens at prediction time:

        1. Compute the distance from the new point to every training point
        2. Find the K closest training points
        3. Take a majority vote among their labels -> that's the prediction

    Key hyperparameter: k (how many neighbors to consider)
        - Small k (e.g. 1): very sensitive to noise, can overfit
        - Large k: smoother decision boundary, can underfit if too large

    Downside: prediction is slow for large datasets, since every single
    prediction requires comparing against ALL training points (this is
    why Day 18 introduces a KD-Tree to speed this up).
    """

    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # "Training" is just memorizing the data
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def _predict_single(self, x):
        # Compute distance from x to every training point
        distances = np.array([euclidean_distance(x, x_train) for x_train in self.X_train])

        # Find indices of the k smallest distances
        k_nearest_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = self.y_train[k_nearest_indices]

        # Majority vote among the k nearest labels
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])

    def predict_proba(self, X):
        """
        Returns the fraction of neighbors voting for each class,
        as a rough probability estimate.
        """
        X = np.array(X)
        classes = np.unique(self.y_train)
        probs = []

        for x in X:
            distances = np.array([euclidean_distance(x, x_train) for x_train in self.X_train])
            k_nearest_indices = np.argsort(distances)[: self.k]
            k_nearest_labels = self.y_train[k_nearest_indices]

            counts = Counter(k_nearest_labels)
            row = [counts.get(c, 0) / self.k for c in classes]
            probs.append(row)

        return np.array(probs)


if __name__ == "__main__":
    # Quick manual test: 2 blobs, some overlap
    np.random.seed(42)
    class0 = np.random.randn(50, 2) + np.array([-1.5, -1.5])
    class1 = np.random.randn(50, 2) + np.array([1.5, 1.5])
    X = np.vstack([class0, class1])
    y = np.array([0] * 50 + [1] * 50)

    model = KNNClassifierScratch(k=5)
    model.fit(X, y)

    preds = model.predict(X)
    accuracy = np.mean(preds == y)
    print("Training accuracy (k=5):", accuracy)