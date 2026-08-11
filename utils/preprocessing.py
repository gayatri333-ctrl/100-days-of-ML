import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True):
    """
    Splits X and y into training and testing sets.

    test_size: fraction of data to reserve for testing (e.g. 0.2 = 20%)
    shuffle: whether to randomly shuffle before splitting (almost always yes,
             unless your data has meaningful order like a time series)
    """
    X = np.array(X)
    y = np.array(y)
    n_samples = X.shape[0]

    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    n_test = int(n_samples * test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


def k_fold_split(X, y, n_splits=5, random_state=None, shuffle=True):
    """
    Generator that yields (X_train, X_val, y_train, y_val) for each fold
    in K-Fold Cross-Validation.

    The idea: split data into K equal-ish chunks ("folds"). Train on
    K-1 folds, validate on the remaining 1 fold. Repeat K times, so
    every fold gets used as the validation set exactly once. This gives
    a much more reliable estimate of model performance than a single
    train/test split, since you're testing on every part of the data.

    Usage:
        for X_train, X_val, y_train, y_val in k_fold_split(X, y, n_splits=5):
            model.fit(X_train, y_train)
            score = evaluate(model, X_val, y_val)
    """
    X = np.array(X)
    y = np.array(y)
    n_samples = X.shape[0]

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop

    for i in range(n_splits):
        val_indices = folds[i]
        train_indices = np.concatenate([folds[j] for j in range(n_splits) if j != i])

        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        yield X_train, X_val, y_train, y_val


def cross_val_score(model_class, X, y, n_splits=5, metric_fn=None, random_state=42, **model_kwargs):
    """
    Runs K-Fold Cross-Validation on a given model class and returns the
    metric score for each fold.

    model_class: the class itself (not an instance), e.g. LinearRegressionScratch
    metric_fn: a function like accuracy or mse from utils/metrics.py
    model_kwargs: any hyperparameters to pass when constructing the model

    Usage:
        from utils.metrics import accuracy
        scores = cross_val_score(LogisticRegressionScratch, X, y,
                                  n_splits=5, metric_fn=accuracy,
                                  learning_rate=0.1, n_iterations=1000)
        print("Mean accuracy:", np.mean(scores))
    """
    if metric_fn is None:
        raise ValueError("You must provide a metric_fn, e.g. accuracy or mse")

    scores = []
    for X_train, X_val, y_train, y_val in k_fold_split(X, y, n_splits=n_splits, random_state=random_state):
        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = metric_fn(y_val, preds)
        scores.append(score)

    return np.array(scores)


# ---------- Feature Scaling ----------

class StandardScaler:
    """
    Standardization (Z-score scaling): rescales each feature to have
    mean = 0 and standard deviation = 1.

        x_scaled = (x - mean) / std

    This is the default choice for most ML algorithms — especially
    gradient descent (converges much faster and more evenly across
    features), and distance-based methods like KNN and SVM.

    IMPORTANT: fit only on training data, then use the SAME mean/std
    to transform both train and test sets. Never fit on test data —
    that would leak test set information into training.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.array(X)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        # avoid divide-by-zero for constant features
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X):
        X = np.array(X)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        X_scaled = np.array(X_scaled)
        return (X_scaled * self.std_) + self.mean_


class MinMaxScaler:
    """
    Normalization (Min-Max scaling): rescales each feature into a fixed
    range, [0, 1] by default.

        x_scaled = (x - min) / (max - min)

    Useful when you need bounded values (e.g. neural network inputs,
    image pixel values), but more sensitive to outliers than
    StandardScaler since a single extreme value stretches the whole range.
    """

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        X = np.array(X)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def transform(self, X):
        X = np.array(X)
        data_range = self.max_ - self.min_
        data_range = np.where(data_range == 0, 1.0, data_range)  # avoid divide-by-zero

        X_std = (X - self.min_) / data_range
        low, high = self.feature_range
        return X_std * (high - low) + low

    def fit_transform(self, X):
        return self.fit(X).transform(X)


if __name__ == "__main__":
    # ---- Train/test split + K-Fold sanity check (from Day 11) ----
    np.random.seed(42)
    X = np.arange(20).reshape(-1, 1)
    y = np.arange(20)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Train size:", len(X_train), "Test size:", len(X_test))

    print("\nK-Fold splits (n_splits=5):")
    for fold_i, (X_tr, X_val, y_tr, y_val) in enumerate(k_fold_split(X, y, n_splits=5, random_state=42)):
        print(f"Fold {fold_i}: train size={len(X_tr)}, val size={len(X_val)}, val indices/values={y_val}")

    # ---- Feature scaling sanity check (Day 12) ----
    print("\n===== Feature Scaling Demo =====")
    X_demo = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0], [5.0, 500.0]])

    std_scaler = StandardScaler()
    X_standardized = std_scaler.fit_transform(X_demo)
    print("Original:\n", X_demo)
    print("Standardized (mean=0, std=1):\n", np.round(X_standardized, 3))
    print("Mean after scaling (should be ~0):", np.round(X_standardized.mean(axis=0), 6))
    print("Std after scaling (should be ~1):", np.round(X_standardized.std(axis=0), 6))

    minmax_scaler = MinMaxScaler()
    X_normalized = minmax_scaler.fit_transform(X_demo)
    print("\nNormalized (range [0, 1]):\n", np.round(X_normalized, 3))