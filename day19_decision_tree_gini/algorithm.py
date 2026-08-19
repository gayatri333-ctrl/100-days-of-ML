import numpy as np


def gini_impurity(y):
    """
    Gini Impurity: measures how "mixed" the classes are in a set of labels.

        Gini = 1 - sum(p_i^2)  for each class i

    Gini = 0   -> perfectly pure (all one class)
    Gini = 0.5 -> maximally impure for binary classification (50/50 split)

    A Decision Tree tries to find splits that REDUCE this impurity as
    much as possible at every step.
    """
    y = np.array(y)
    if len(y) == 0:
        return 0
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)


class TreeNode:
    """A single node in the Decision Tree — either a decision (split) or a leaf (prediction)."""
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # which feature this node splits on
        self.threshold = threshold          # the split point (feature <= threshold -> left)
        self.left = left                    # left subtree (feature <= threshold)
        self.right = right                  # right subtree (feature > threshold)
        self.value = value                  # only set for LEAF nodes: the predicted class

    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifierScratch:
    """
    Decision Tree Classifier using Gini Impurity to choose splits.

    The algorithm (recursive, greedy):
        1. For the current data, try every possible (feature, threshold) split
        2. Pick the split that gives the BIGGEST reduction in Gini impurity
           (i.e. makes the two resulting groups as pure as possible)
        3. Recursively repeat on each resulting group
        4. Stop when: max_depth reached, a node is already pure, or too
           few samples remain to split further

    At prediction time, a new point just "walks down" the tree, following
    left/right based on each node's feature/threshold check, until it
    hits a leaf — that leaf's stored class is the prediction.
    """

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _best_split(self, X, y):
        """
        Searches all features and all possible thresholds to find the
        split that minimizes weighted Gini impurity of the two children.
        """
        n_samples, n_features = X.shape
        best_gini = float("inf")
        best_feature, best_threshold = None, None

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue  # skip splits that don't actually divide the data

                y_left, y_right = y[left_mask], y[right_mask]

                # Weighted average of the two children's Gini impurities
                n_left, n_right = len(y_left), len(y_right)
                weighted_gini = (n_left / n_samples) * gini_impurity(y_left) + \
                                 (n_right / n_samples) * gini_impurity(y_right)

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold, best_gini

    def _build_tree(self, X, y, depth=0):
        n_samples = len(y)
        n_classes = len(np.unique(y))

        # Stopping conditions -> create a leaf node
        if (depth >= self.max_depth or
                n_classes == 1 or
                n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)

        feature_index, threshold, gini = self._best_split(X, y)

        # If no valid split improves things, also make a leaf
        if feature_index is None:
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)

        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(feature_index=feature_index, threshold=threshold,
                         left=left_subtree, right=right_subtree)

    def _most_common_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _predict_single(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_single(x, self.root) for x in X])


if __name__ == "__main__":
    # Quick manual test: 2 blobs
    np.random.seed(42)
    class0 = np.random.randn(50, 2) + np.array([-2, -2])
    class1 = np.random.randn(50, 2) + np.array([2, 2])
    X = np.vstack([class0, class1])
    y = np.array([0] * 50 + [1] * 50)

    model = DecisionTreeClassifierScratch(max_depth=5)
    model.fit(X, y)

    preds = model.predict(X)
    accuracy = np.mean(preds == y)
    print("Training accuracy:", accuracy)