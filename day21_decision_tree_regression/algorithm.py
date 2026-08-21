import numpy as np


def variance(y):
    """
    Variance: measures how spread out a set of continuous target values is.

    This plays the same role for REGRESSION trees that Gini/Entropy played
    for CLASSIFICATION trees (Days 19-20) — it's the "impurity" measure
    we're trying to reduce with each split.

    Low variance -> the values in this group are similar to each other
                     (good leaf, confident prediction)
    High variance -> the values are spread out (needs more splitting)
    """
    y = np.array(y)
    if len(y) == 0:
        return 0
    return np.var(y)


def variance_reduction(y_parent, y_left, y_right):
    """
    How much variance DECREASES after a split — the regression equivalent
    of Information Gain (Day 20).

        reduction = Var(parent) - weighted_average(Var(children))

    Higher reduction = better split (children are more internally
    consistent than the parent was).
    """
    n = len(y_parent)
    n_left, n_right = len(y_left), len(y_right)

    if n_left == 0 or n_right == 0:
        return 0

    weighted_child_variance = (n_left / n) * variance(y_left) + (n_right / n) * variance(y_right)
    return variance(y_parent) - weighted_child_variance


class TreeNode:
    """Same node structure as Days 19-20."""
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # for regression, this is a continuous number (not a class)

    def is_leaf(self):
        return self.value is not None


class DecisionTreeRegressorScratch:
    """
    Decision Tree Regressor — predicts CONTINUOUS values instead of classes.

    The tree-building algorithm is structurally IDENTICAL to Days 19-20:
    recursively find the best (feature, threshold) split, build left/right
    subtrees, stop at some condition. Only two things change:

        1. Split quality is measured by VARIANCE REDUCTION, not Gini/Entropy
        2. Leaf nodes store the MEAN of their group's y-values, not the
           majority class

    This is the same "swap one piece of a shared framework" pattern you
    saw going from Gini (Day 19) to Entropy (Day 20).
    """

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_reduction = -1
        best_feature, best_threshold = None, None

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]
                reduction = variance_reduction(y, y_left, y_right)

                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold, best_reduction

    def _build_tree(self, X, y, depth=0):
        n_samples = len(y)

        # Stopping conditions -> create a leaf storing the MEAN value
        if (depth >= self.max_depth or
                n_samples < self.min_samples_split or
                variance(y) == 0):  # already perfectly consistent
            leaf_value = np.mean(y)
            return TreeNode(value=leaf_value)

        feature_index, threshold, reduction = self._best_split(X, y)

        if feature_index is None or reduction <= 0:
            leaf_value = np.mean(y)
            return TreeNode(value=leaf_value)

        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(feature_index=feature_index, threshold=threshold,
                         left=left_subtree, right=right_subtree)

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
    # Quick manual test: y = sin(x), same non-linear curve as Day 17
    np.random.seed(42)
    X = np.sort(np.random.rand(100, 1) * 10, axis=0)
    y = np.sin(X[:, 0]) + np.random.randn(100) * 0.1

    model = DecisionTreeRegressorScratch(max_depth=5)
    model.fit(X, y)

    preds = model.predict(X)
    mse = np.mean((y - preds) ** 2)
    print("Training MSE (max_depth=5):", mse)