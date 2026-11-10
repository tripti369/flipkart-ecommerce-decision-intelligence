"""Shared helper routines for Flipkart E-commerce Decision Intelligence capstone.
Contains data loading, stratified splitting, standardisation, model routines, and metrics.
"""

import csv
import math
import numpy as np

DATA_PATH = "data/records.csv"

FEATURE_COLS = [
    "customer_tenure_months",
    "complaint_count_90d",
    "discount_order_share",
    "avg_order_value_inr_hundreds",
    "loyalty_score",
    "cod_flag",
    "plus_member_flag",
]

CONTINUOUS_COLS = [
    "customer_tenure_months",
    "complaint_count_90d",
    "discount_order_share",
    "avg_order_value_inr_hundreds",
    "loyalty_score",
]


def load_records(path=DATA_PATH):
    """Load records.csv as a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_stratified_split(rows):
    """Perform the pinned 70/30 stratified train/test split preserving file order.
    
    Rows with outcome 0: first floor(7 * count / 10) to train, rest to test.
    Rows with outcome 1: first floor(7 * count / 10) to train, rest to test.
    Both train and test maintain original file order.
    """
    neg_indices = [i for i, r in enumerate(rows) if int(r["churned"]) == 0]
    pos_indices = [i for i, r in enumerate(rows) if int(r["churned"]) == 1]

    n_train_neg = (7 * len(neg_indices)) // 10
    n_train_pos = (7 * len(pos_indices)) // 10

    train_indices = sorted(neg_indices[:n_train_neg] + pos_indices[:n_train_pos])
    test_indices = sorted(neg_indices[n_train_neg:] + pos_indices[n_train_pos:])

    train_rows = [rows[i] for i in train_indices]
    test_rows = [rows[i] for i in test_indices]

    return train_indices, test_indices, train_rows, test_rows


def standardise_features(train_arr, test_arr=None):
    """Standardise features using training mean and sample SD (ddof=1)."""
    means = np.mean(train_arr, axis=0)
    sds = np.std(train_arr, axis=0, ddof=1)
    train_std = (train_arr - means) / sds
    if test_arr is not None:
        test_std = (test_arr - means) / sds
        return train_std, test_std, means, sds
    return train_std, means, sds


def sigmoid(z):
    """Compute logistic sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def bce_loss(y_true, y_prob):
    """Binary cross-entropy loss with numerical stabilization."""
    eps = 1e-15
    p = np.clip(y_prob, eps, 1.0 - eps)
    return -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))


def compute_confusion_and_metrics(y_true, y_prob, tau=0.5):
    """Compute confusion matrix cells and evaluation metrics at threshold tau.
    
    Positive prediction when probability >= tau.
    """
    pred = (y_prob >= tau).astype(int)
    y_true_int = y_true.astype(int)

    tp = int(np.sum((pred == 1) & (y_true_int == 1)))
    fp = int(np.sum((pred == 1) & (y_true_int == 0)))
    tn = int(np.sum((pred == 0) & (y_true_int == 0)))
    fn = int(np.sum((pred == 0) & (y_true_int == 1)))

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "fpr": fpr,
    }


def fit_logistic_regression(X_train, y_train, epochs=1000, eta=0.1):
    """Fit logistic regression model using gradient descent."""
    n_samples = len(y_train)
    n_features = X_train.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    losses = {}

    losses[0] = bce_loss(y_train, sigmoid(X_train @ w + b))

    for ep in range(1, epochs + 1):
        z = X_train @ w + b
        y_hat = sigmoid(z)
        dw = (1.0 / n_samples) * (X_train.T @ (y_hat - y_train))
        db = (1.0 / n_samples) * np.sum(y_hat - y_train)
        w -= eta * dw
        b -= eta * db
        if ep in [1, 10, 100, 1000]:
            losses[ep] = bce_loss(y_train, sigmoid(X_train @ w + b))

    return w, b, losses


def knn_predict(X_ref, y_ref, X_query, K):
    """Predict query classes and probabilities using K-nearest neighbours.
    
    Tie breaking by reference row position in file order via stable sort.
    """
    preds = []
    probs = []
    for q in X_query:
        dists = np.sqrt(np.sum((X_ref - q) ** 2, axis=1))
        ranked_idx = np.argsort(dists, kind="stable")[:K]
        pos_count = int(np.sum(y_ref[ranked_idx]))
        pred = 1 if pos_count >= (K + 1) // 2 else 0
        prob = pos_count / K
        preds.append(pred)
        probs.append(prob)
    return np.array(preds), np.array(probs)
