import numpy as np


# ---------- Regression Metrics ----------

def mse(y_true, y_pred):
    """Mean Squared Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    """Mean Absolute Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """R-squared (coefficient of determination)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


# ---------- Classification Metrics ----------

def accuracy(y_true, y_pred):
    """Accuracy = correct predictions / total predictions"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    """
    Returns confusion matrix as a 2x2 numpy array for binary classification:
    [[TN, FP],
     [FN, TP]]
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])


def precision(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


if __name__ == "__main__":
    # quick sanity check
    y_true = [1, 0, 1, 1, 0]
    y_pred = [1, 0, 0, 1, 0]
    print("Accuracy:", accuracy(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Precision:", precision(y_true, y_pred))
    print("Recall:", recall(y_true, y_pred))
    print("F1:", f1_score(y_true, y_pred))