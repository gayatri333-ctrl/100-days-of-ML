import numpy as np


def entropy(y):
    """
    Entropy: another way (besides Gini) to measure how "mixed" a set
    of labels is, borrowed from information theory.

        Entropy = -sum(p_i * log2(p_i))  for each class i

    Entropy = 0   -> perfectly pure (all one class)
    Entropy = 1.0 -> maximally impure for binary classification (50/50 split)

    Interpretation: entropy measures the "surprise" or "uncertainty" in
    the label distribution. A pure node has zero surprise (you already
    know the answer). A 50/50 split has maximum surprise.
    """
    y = np.array(y)
    if len(y) == 0:
        return 0
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    # add tiny epsilon to avoid log2(0)
    return -np.sum(probabilities * np.log2(probabilities + 1e-15))


def information_gain(y_parent, y_left, y_right):
    """
    Information Gain: how much entropy DECREASES after a split.

        IG = Entropy(parent) - weighted_average(Entropy(children))

    Higher IG = better split (bigger reduction in uncertainty).
    This is the entropy-based equivalent of "weighted Gini reduction"
    from Day 19 — same goal, different underlying math.
    """
    n = len(y_parent)
    n_left, n_right = len(y_left), len(y_right)

    if n_left == 0 or n_right == 0:
        return 0

    weighted_child_entropy = (n_left / n) * entropy(y_left) + (n_right / n) * entropy(y_right)
    return entropy(y_parent) - weighted_child_entropy


class TreeNode:
    """Same node structure as Day 19 — either a decision split or a leaf prediction."""
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTreeEntropyScratch:
    """
    Decision Tree Classifier using Entropy + Information Gain to choose
    splits, instead of Gini impurity (Day 19).

    The tree-building algorithm itself is IDENTICAL to Day 19 — recursive,
    greedy splitting. The only thing that changes is which split gets
    picked as "best": here we MAXIMIZE information gain instead of
    MINIMIZING weighted Gini. In practice, both criteria tend to produce
    very similar trees on most datasets; entropy is slightly more
    computationally expensive (due to the log2 calls) but is grounded in
    information theory, which some fields prefer for interpretability.
    """

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gain = -1  # we want to MAXIMIZE gain, so start low
        best_feature, best_threshold = None, None

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]
                gain = information_gain(y, y_left, y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth=0):
        n_samples = len(y)
        n_classes = len(np.unique(y))

        if (depth >= self.max_depth or
                n_classes == 1 or
                n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)

        feature_index, threshold, gain = self._best_split(X, y)

        # If no split provides any information gain, make a leaf
        if feature_index is None or gain <= 0:
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

    model = DecisionTreeEntropyScratch(max_depth=5)
    model.fit(X, y)

    preds = model.predict(X)
    accuracy = np.mean(preds == y)
    print("Training accuracy:", accuracy)

    # Quick entropy sanity checks
    print("\nEntropy of pure set [1,1,1,1]:", entropy([1, 1, 1, 1]))
    print("Entropy of 50/50 set [0,1,0,1]:", entropy([0, 1, 0, 1]))