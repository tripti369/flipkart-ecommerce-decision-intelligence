"""Part 4 — Model-choice and architecture memo.
Produces:
  outputs/part4_results.txt
  outputs/part4_splits.csv
  outputs/part4_voting.csv
  outputs/part4_adaboost.csv
  outputs/part4_architecture.csv
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
    compute_confusion_and_metrics,
    fit_logistic_regression,
    knn_predict,
    select_best_knn_k,
)


def run_part4():
    rows = load_records()
    train_idx, test_idx, train_rows, test_rows = get_stratified_split(rows)

    # 4.1 Candidate splits on 560 training rows in original units
    n_train = len(train_rows)
    pos_train = sum(int(r["churned"]) for r in train_rows)

    def entropy(pos, total):
        if total == 0 or pos == 0 or pos == total:
            return 0.0
        p1 = pos / total
        p0 = 1.0 - p1
        return -(p1 * math.log2(p1) + p0 * math.log2(p0))

    def gini(pos, total):
        if total == 0:
            return 0.0
        p1 = pos / total
        p0 = 1.0 - p1
        return 1.0 - (p1 ** 2 + p0 ** 2)

    root_entropy = entropy(pos_train, n_train)
    root_gini = gini(pos_train, n_train)

    candidate_splits = [
        ("S1", lambda r: float(r["cod_flag"]) == 1.0),
        ("S2", lambda r: float(r["plus_member_flag"]) == 1.0),
        ("S3", lambda r: float(r["discount_order_share"]) > 0.5),
        ("S4", lambda r: float(r["customer_tenure_months"]) <= 24.0),
    ]

    split_results = []
    best_ig = -1.0
    best_gg = -1.0
    best_split_by_ig = None
    best_split_by_gg = None
    best_split_cond = None

    for name, cond in candidate_splits:
        yes_rows = [r for r in train_rows if cond(r)]
        no_rows = [r for r in train_rows if not cond(r)]
        n_yes = len(yes_rows)
        pos_yes = sum(int(r["churned"]) for r in yes_rows)
        n_no = len(no_rows)
        pos_no = sum(int(r["churned"]) for r in no_rows)

        child_entropy = (n_yes / n_train) * entropy(pos_yes, n_yes) + (n_no / n_train) * entropy(pos_no, n_no)
        child_gini = (n_yes / n_train) * gini(pos_yes, n_yes) + (n_no / n_train) * gini(pos_no, n_no)
        ig = root_entropy - child_entropy
        gg = root_gini - child_gini

        split_results.append((name, n_yes, pos_yes, n_no, pos_no, ig, gg))

        if ig > best_ig:
            best_ig = ig
            best_split_by_ig = name
            best_split_cond = cond
        if gg > best_gg:
            best_gg = gg
            best_split_by_gg = name

    os.makedirs("outputs", exist_ok=True)
    with open(os.path.join("outputs", "part4_splits.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"])
        for r in split_results:
            writer.writerow([r[0], r[1], r[2], r[3], r[4], f"{r[5]:.4f}", f"{r[6]:.4f}"])

    # 4.2 Stump and Voting models
    # The stump is dynamically determined from the best split by information gain
    yes_train_stump = [r for r in train_rows if best_split_cond(r)]
    no_train_stump = [r for r in train_rows if not best_split_cond(r)]
    stump_rate_yes = sum(int(r["churned"]) for r in yes_train_stump) / len(yes_train_stump)
    stump_rate_no = sum(int(r["churned"]) for r in no_train_stump) / len(no_train_stump)

    # Prepare features for logistic and knn
    X_train_raw = np.array([[float(r[col]) for col in FEATURE_COLS] for r in train_rows])
    y_train = np.array([float(r["churned"]) for r in train_rows])
    X_test_raw = np.array([[float(r[col]) for col in FEATURE_COLS] for r in test_rows])
    y_test = np.array([float(r["churned"]) for r in test_rows])

    X_train_std, X_test_std, _, _ = standardise_features(X_train_raw, X_test_raw)

    # 1. Logistic model (1000 updates, eta=0.1)
    w_lr, b_lr, _ = fit_logistic_regression(X_train_std, y_train, epochs=1000, eta=0.1)
    prob_logistic = sigmoid(X_test_std @ w_lr + b_lr)
    pred_logistic = (prob_logistic >= 0.3).astype(int)

    # 2. KNN model (K dynamically chosen via cross-validation on training data)
    best_k, _ = select_best_knn_k(X_train_std, y_train, len(train_rows))
    pred_knn, prob_knn = knn_predict(X_train_std, y_train, X_test_std, best_k)

    # 3. Stump model (evaluated dynamically using best_split_cond)
    prob_stump = np.array([
        stump_rate_yes if best_split_cond(r) else stump_rate_no
        for r in test_rows
    ])
    pred_stump = (prob_stump >= 0.3).astype(int)

    # 4. Hard vote: positive when at least two of the three above predict positive
    pred_hard = ((pred_logistic + pred_knn + pred_stump) >= 2).astype(int)

    # 5. Soft vote: positive when mean of the three probabilities is >= 0.3
    prob_mean = (prob_logistic + prob_knn + prob_stump) / 3.0
    pred_soft = (prob_mean >= 0.3).astype(int)

    model_evals = [
        ("logistic", pred_logistic),
        ("knn", pred_knn),
        ("stump", pred_stump),
        ("hard_vote", pred_hard),
        ("soft_vote", pred_soft),
    ]

    voting_csv_rows = []
    for model_name, p_class in model_evals:
        m = compute_confusion_and_metrics(y_test, p_class, tau=0.5)
        voting_csv_rows.append([
            model_name,
            m["tp"],
            m["fp"],
            m["tn"],
            m["fn"],
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1']:.4f}",
        ])

    with open(os.path.join("outputs", "part4_voting.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"])
        writer.writerows(voting_csv_rows)

    # 4.3 Two rounds of AdaBoost by hand
    y_ada = np.array([1, 1, 1, -1, -1, -1, 1, -1])
    h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
    h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])

    adaboost_csv_rows = []

    # Round 1
    w_before_1 = np.full(8, 1.0 / 8.0)
    eps1 = float(np.sum(w_before_1[h1 != y_ada]))
    alpha1 = 0.5 * math.log((1.0 - eps1) / eps1)
    w_unnorm_1 = w_before_1 * np.exp(-alpha1 * y_ada * h1)
    w_after_1 = w_unnorm_1 / np.sum(w_unnorm_1)

    for i in range(8):
        adaboost_csv_rows.append([1, i + 1, f"{w_before_1[i]:.4f}", f"{w_unnorm_1[i]:.4f}", f"{w_after_1[i]:.4f}"])

    # Round 2
    w_before_2 = w_after_1.copy()
    eps2 = float(np.sum(w_before_2[h2 != y_ada]))
    alpha2 = 0.5 * math.log((1.0 - eps2) / eps2)
    w_unnorm_2 = w_before_2 * np.exp(-alpha2 * y_ada * h2)
    w_after_2 = w_unnorm_2 / np.sum(w_unnorm_2)

    for i in range(8):
        adaboost_csv_rows.append([2, i + 1, f"{w_before_2[i]:.4f}", f"{w_unnorm_2[i]:.4f}", f"{w_after_2[i]:.4f}"])

    with open(os.path.join("outputs", "part4_adaboost.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "point", "weight_before", "weight_unnormalised", "weight_after"])
        writer.writerows(adaboost_csv_rows)

    ensemble_pred = np.where((alpha1 * h1 + alpha2 * h2) >= 0, 1, -1)
    ensemble_accuracy = float(np.mean(ensemble_pred == y_ada))

    # 4.4 Out-of-bag share
    oob_fraction = (1.0 - 1.0 / 560.0) ** 560.0

    # 4.5 Neural network sizing
    # Model A Option 1 MLP: 784 -> 16 -> 16 -> 10
    mlp_h1 = 784 * 16 + 16
    mlp_h2 = 16 * 16 + 16
    mlp_out = 16 * 10 + 10
    mlp_total = mlp_h1 + mlp_h2 + mlp_out

    # Model A Option 2 CNN:
    cnn_c1 = (5 * 5 * 1 + 1) * 6
    cnn_c2 = (5 * 5 * 6 + 1) * 16
    cnn_d1 = (5 * 5 * 16) * 32 + 32
    cnn_d2 = 32 * 10 + 10
    cnn_total = cnn_c1 + cnn_c2 + cnn_d1 + cnn_d2

    architecture_rows = [
        ["mlp_hidden1", mlp_h1],
        ["mlp_hidden2", mlp_h2],
        ["mlp_output", mlp_out],
        ["cnn_conv1", cnn_c1],
        ["cnn_conv2", cnn_c2],
        ["cnn_dense1", cnn_d1],
        ["cnn_dense2", cnn_d2],
    ]

    with open(os.path.join("outputs", "part4_architecture.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "parameters"])
        writer.writerows(architecture_rows)

    # Scaling comparison
    fc_layer_params = (128 * 128 * 3) * 1024 + 1024
    conv_layer_params = (3 * 3 * 3 * 64) + 64

    # Model B RNN
    rnn_total = (128 * 64) + (128 * 128) + 128 + (16 * 128) + 16

    # Results file
    results = [
        ("root_entropy", f"{root_entropy:.4f}"),
        ("root_gini", f"{root_gini:.4f}"),
        ("best_split_by_info_gain", f"{best_split_by_ig}"),
        ("best_split_by_gini_gain", f"{best_split_by_gg}"),
        ("stump_rate_yes", f"{stump_rate_yes:.4f}"),
        ("stump_rate_no", f"{stump_rate_no:.4f}"),
        ("round1_error", f"{eps1:.4f}"),
        ("round1_alpha", f"{alpha1:.4f}"),
        ("round2_error", f"{eps2:.4f}"),
        ("round2_alpha", f"{alpha2:.4f}"),
        ("ensemble_accuracy", f"{ensemble_accuracy:.4f}"),
        ("oob_fraction", f"{oob_fraction:.4f}"),
        ("mlp_total", f"{mlp_total}"),
        ("cnn_total", f"{cnn_total}"),
        ("fc_layer_params", f"{fc_layer_params}"),
        ("conv_layer_params", f"{conv_layer_params}"),
        ("rnn_total", f"{rnn_total}"),
    ]

    with open(os.path.join("outputs", "part4_results.txt"), "w", encoding="utf-8") as f:
        for k, v in results:
            f.write(f"{k} = {v}\n")

    print(f"Part 4 complete: {len(results)} keys written to outputs/part4_results.txt")


if __name__ == "__main__":
    run_part4()
