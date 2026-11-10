"""Part 2 — Classification and evaluation.
Produces:
  outputs/part2_results.txt
  outputs/part2_thresholds.csv
  outputs/part2_knn_cv.csv
"""

import csv
import math
import os
import numpy as np
from common import (
    load_records,
    get_stratified_split,
    standardise_features,
    FEATURE_COLS,
    sigmoid,
    bce_loss,
    compute_confusion_and_metrics,
    fit_logistic_regression,
    knn_predict,
)


def run_part2():
    rows = load_records()
    train_idx, test_idx, train_rows, test_rows = get_stratified_split(rows)

    train_rows_count = len(train_rows)
    test_rows_count = len(test_rows)
    train_positives = sum(int(r["churned"]) for r in train_rows)
    test_positives = sum(int(r["churned"]) for r in test_rows)

    X_train_raw = np.array([[float(r[col]) for col in FEATURE_COLS] for r in train_rows])
    y_train = np.array([float(r["churned"]) for r in train_rows])
    X_test_raw = np.array([[float(r[col]) for col in FEATURE_COLS] for r in test_rows])
    y_test = np.array([float(r["churned"]) for r in test_rows])

    X_train, X_test, _, _ = standardise_features(X_train_raw, X_test_raw)

    # 2.2 Logistic regression from scratch (1000 updates, eta=0.1)
    w, b, losses = fit_logistic_regression(X_train, y_train, epochs=1000, eta=0.1)

    test_bce = bce_loss(y_test, sigmoid(X_test @ w + b))
    test_probs = sigmoid(X_test @ w + b)
    train_probs = sigmoid(X_train @ w + b)

    # 2.3 Learning-rate effect (200 updates)
    n_tr = len(y_train)
    eta_losses = {}
    for test_eta in [0.01, 0.1, 1.0]:
        w_tmp = np.zeros(7)
        b_tmp = 0.0
        for _ in range(200):
            z = X_train @ w_tmp + b_tmp
            y_hat = sigmoid(z)
            dw = (1.0 / n_tr) * (X_train.T @ (y_hat - y_train))
            db = (1.0 / n_tr) * np.sum(y_hat - y_train)
            w_tmp -= test_eta * dw
            b_tmp -= test_eta * db
        eta_losses[test_eta] = bce_loss(y_train, sigmoid(X_train @ w_tmp + b_tmp))

    # 2.4 Odds ratios
    odds_ratios = [math.exp(w[j]) for j in range(7)]

    # 2.5 Evaluate at tau=0.5 and tau=0.3
    m_05 = compute_confusion_and_metrics(y_test, test_probs, tau=0.5)
    m_03 = compute_confusion_and_metrics(y_test, test_probs, tau=0.3)
    blind_accuracy = np.sum(y_test == 0) / len(y_test)

    # 2.6 Threshold sweep
    os.makedirs("outputs", exist_ok=True)
    sweep_rows = []
    for k in range(1, 10):
        tau = round(k * 0.1, 1)
        m_tau = compute_confusion_and_metrics(y_test, test_probs, tau=tau)
        sweep_rows.append([
            f"{tau:.4f}",
            m_tau["tp"],
            m_tau["fp"],
            m_tau["tn"],
            m_tau["fn"],
            f"{m_tau['recall']:.4f}",  # tpr = recall
            f"{m_tau['fpr']:.4f}",
        ])

    with open(os.path.join("outputs", "part2_thresholds.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"])
        writer.writerows(sweep_rows)

    # 2.7 K-nearest neighbours with 5-fold cross-validation
    fold_assignments = np.array([i % 5 for i in range(len(train_idx))])
    K_list = list(range(1, 23, 2))
    cv_records = []
    best_k = None
    best_cv_f1 = -1.0

    for K in K_list:
        fold_f1s = []
        for f in range(5):
            val_mask = (fold_assignments == f)
            ref_mask = ~val_mask
            X_ref, y_ref = X_train[ref_mask], y_train[ref_mask]
            X_val, y_val = X_train[val_mask], y_train[val_mask]
            pred_val, _ = knn_predict(X_ref, y_ref, X_val, K)
            metrics_fold = compute_confusion_and_metrics(y_val, pred_val, tau=0.5)
            fold_f1s.append(metrics_fold["f1"])
        mean_f1 = float(np.mean(fold_f1s))
        cv_records.append((K, mean_f1))
        if mean_f1 > best_cv_f1:  # ties go to smaller K automatically because K is strictly increasing
            best_cv_f1 = mean_f1
            best_k = K

    with open(os.path.join("outputs", "part2_knn_cv.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "mean_cv_f1"])
        for K_val, score in cv_records:
            writer.writerow([K_val, f"{score:.4f}"])

    test_pred_knn, _ = knn_predict(X_train, y_train, X_test, best_k)
    m_knn = compute_confusion_and_metrics(y_test, test_pred_knn, tau=0.5)

    # 2.8 Alert threshold with money
    best_cost_tau = None
    lowest_train_cost = float("inf")
    for k in range(1, 20):
        tau = k / 20.0
        m_cost_tr = compute_confusion_and_metrics(y_train, train_probs, tau=tau)
        cost_tr = 50 * m_cost_tr["fp"] + 400 * m_cost_tr["fn"]
        if cost_tr < lowest_train_cost:
            lowest_train_cost = cost_tr
            best_cost_tau = tau

    cost_tau = best_cost_tau
    cost_train_at_tau = lowest_train_cost

    m_cost_test_tau = compute_confusion_and_metrics(y_test, test_probs, tau=cost_tau)
    cost_test_at_tau = 50 * m_cost_test_tau["fp"] + 400 * m_cost_test_tau["fn"]

    m_cost_test_05 = compute_confusion_and_metrics(y_test, test_probs, tau=0.5)
    cost_test_at_05 = 50 * m_cost_test_05["fp"] + 400 * m_cost_test_05["fn"]

    cost_test_flag_nobody = 400 * int(np.sum(y_test == 1))

    # Write part2_results.txt
    results = [
        ("train_rows", f"{train_rows_count}"),
        ("train_positives", f"{train_positives}"),
        ("test_rows", f"{test_rows_count}"),
        ("test_positives", f"{test_positives}"),
        ("loss_epoch_0", f"{losses[0]:.4f}"),
        ("loss_epoch_1", f"{losses[1]:.4f}"),
        ("loss_epoch_10", f"{losses[10]:.4f}"),
        ("loss_epoch_100", f"{losses[100]:.4f}"),
        ("loss_epoch_1000", f"{losses[1000]:.4f}"),
        ("weight_X1", f"{w[0]:.4f}"),
        ("weight_X2", f"{w[1]:.4f}"),
        ("weight_X3", f"{w[2]:.4f}"),
        ("weight_X4", f"{w[3]:.4f}"),
        ("weight_X5", f"{w[4]:.4f}"),
        ("weight_X6", f"{w[5]:.4f}"),
        ("weight_X7", f"{w[6]:.4f}"),
        ("bias", f"{b:.4f}"),
        ("test_bce", f"{test_bce:.4f}"),
        ("loss_epoch_200_eta_0.01", f"{eta_losses[0.01]:.4f}"),
        ("loss_epoch_200_eta_0.1", f"{eta_losses[0.1]:.4f}"),
        ("loss_epoch_200_eta_1.0", f"{eta_losses[1.0]:.4f}"),
        ("odds_ratio_X1", f"{odds_ratios[0]:.4f}"),
        ("odds_ratio_X2", f"{odds_ratios[1]:.4f}"),
        ("odds_ratio_X3", f"{odds_ratios[2]:.4f}"),
        ("odds_ratio_X4", f"{odds_ratios[3]:.4f}"),
        ("odds_ratio_X5", f"{odds_ratios[4]:.4f}"),
        ("odds_ratio_X6", f"{odds_ratios[5]:.4f}"),
        ("odds_ratio_X7", f"{odds_ratios[6]:.4f}"),
        ("tau_0.5_tp", f"{m_05['tp']}"),
        ("tau_0.5_fp", f"{m_05['fp']}"),
        ("tau_0.5_tn", f"{m_05['tn']}"),
        ("tau_0.5_fn", f"{m_05['fn']}"),
        ("tau_0.5_accuracy", f"{m_05['accuracy']:.4f}"),
        ("tau_0.5_precision", f"{m_05['precision']:.4f}"),
        ("tau_0.5_recall", f"{m_05['recall']:.4f}"),
        ("tau_0.5_f1", f"{m_05['f1']:.4f}"),
        ("tau_0.5_specificity", f"{m_05['specificity']:.4f}"),
        ("tau_0.3_tp", f"{m_03['tp']}"),
        ("tau_0.3_fp", f"{m_03['fp']}"),
        ("tau_0.3_tn", f"{m_03['tn']}"),
        ("tau_0.3_fn", f"{m_03['fn']}"),
        ("tau_0.3_accuracy", f"{m_03['accuracy']:.4f}"),
        ("tau_0.3_precision", f"{m_03['precision']:.4f}"),
        ("tau_0.3_recall", f"{m_03['recall']:.4f}"),
        ("tau_0.3_f1", f"{m_03['f1']:.4f}"),
        ("tau_0.3_specificity", f"{m_03['specificity']:.4f}"),
        ("blind_accuracy", f"{blind_accuracy:.4f}"),
        ("knn_best_k", f"{best_k}"),
        ("knn_tp", f"{m_knn['tp']}"),
        ("knn_fp", f"{m_knn['fp']}"),
        ("knn_tn", f"{m_knn['tn']}"),
        ("knn_fn", f"{m_knn['fn']}"),
        ("knn_accuracy", f"{m_knn['accuracy']:.4f}"),
        ("knn_precision", f"{m_knn['precision']:.4f}"),
        ("knn_recall", f"{m_knn['recall']:.4f}"),
        ("knn_f1", f"{m_knn['f1']:.4f}"),
        ("cost_tau", f"{cost_tau:.4f}"),
        ("cost_train_at_tau", f"{cost_train_at_tau}"),
        ("cost_test_at_tau", f"{cost_test_at_tau}"),
        ("cost_test_at_0.5", f"{cost_test_at_05}"),
        ("cost_test_flag_nobody", f"{cost_test_flag_nobody}"),
    ]

    with open(os.path.join("outputs", "part2_results.txt"), "w", encoding="utf-8") as f:
        for k, v in results:
            f.write(f"{k} = {v}\n")

    print(f"Part 2 complete: {len(results)} keys written to outputs/part2_results.txt")


if __name__ == "__main__":
    run_part2()
