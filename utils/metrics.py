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


# ---------- Binary Classification Metrics ----------

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


# ---------- Multiclass Classification Metrics ----------

def confusion_matrix_multiclass(y_true, y_pred, n_classes=None):
    """
    General confusion matrix for ANY number of classes.
    Rows = true class, Columns = predicted class.
    cm[i][j] = number of samples with true label i predicted as label j.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1
    return cm


def precision_recall_f1_multiclass(y_true, y_pred, average="macro"):
    """
    Precision, Recall, and F1 for multiclass classification.

    average="macro" -> compute per-class then take unweighted mean
                        (treats every class equally, regardless of size)
    average="micro" -> pool all TP/FP/FN across classes, then compute once
                        (equivalent to accuracy when every sample gets exactly
                        one predicted label)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))

    precisions, recalls, f1s = [], [], []
    total_tp, total_fp, total_fn = 0, 0, 0

    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        total_tp += tp
        total_fp += fp
        total_fn += fn

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    if average == "macro":
        return np.mean(precisions), np.mean(recalls), np.mean(f1s)
    elif average == "micro":
        p_micro = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        r_micro = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro) if (p_micro + r_micro) > 0 else 0.0
        return p_micro, r_micro, f1_micro
    else:
        raise ValueError("average must be 'macro' or 'micro'")


# ---------- ROC-AUC (Binary Classification) ----------

def roc_curve(y_true, y_scores):
    """
    Computes the ROC curve points: False Positive Rate (FPR) vs
    True Positive Rate (TPR) at every possible probability threshold.

    y_scores must be predicted PROBABILITIES (not hard 0/1 labels).
    Returns (fpr_array, tpr_array, thresholds_array).
    """
    y_true, y_scores = np.array(y_true), np.array(y_scores)

    thresholds = np.unique(y_scores)
    thresholds = np.sort(thresholds)[::-1]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    tpr_list = []
    fpr_list = []

    for thresh in thresholds:
        y_pred_at_thresh = (y_scores >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred_at_thresh == 1))
        fp = np.sum((y_true == 0) & (y_pred_at_thresh == 1))

        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return np.array(fpr_list), np.array(tpr_list), thresholds


def roc_auc_score(y_true, y_scores):
    """
    Area Under the ROC Curve — summarizes model performance across ALL
    thresholds into one number. 0.5 = random guessing, 1.0 = perfect.

    Computed here using the trapezoidal rule over the ROC curve points.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    order = np.argsort(fpr)
    fpr_sorted = fpr[order]
    tpr_sorted = tpr[order]

    return np.trapz(tpr_sorted, fpr_sorted)


if __name__ == "__main__":
    # ---- Binary classification sanity check ----
    y_true = [1, 0, 1, 1, 0]
    y_pred = [1, 0, 0, 1, 0]
    print("Accuracy:", accuracy(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Precision:", precision(y_true, y_pred))
    print("Recall:", recall(y_true, y_pred))
    print("F1:", f1_score(y_true, y_pred))

    # ---- Multiclass sanity check ----
    y_true_mc = [0, 1, 2, 2, 1, 0]
    y_pred_mc = [0, 2, 2, 2, 1, 1]
    print("\nMulticlass Confusion Matrix:\n", confusion_matrix_multiclass(y_true_mc, y_pred_mc))
    p, r, f1 = precision_recall_f1_multiclass(y_true_mc, y_pred_mc, average="macro")
    print(f"Macro Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

    # ---- ROC-AUC sanity check ----
    y_true_roc = [0, 0, 1, 1]
    y_scores_roc = [0.1, 0.4, 0.35, 0.8]
    auc = roc_auc_score(y_true_roc, y_scores_roc)
    print(f"\nROC-AUC: {auc:.4f}")