"""Part 3 — Segmentation and dimensionality reduction.
Produces:
  outputs/part3_results.txt
  outputs/part3_segments.csv
  outputs/part3_hierarchical.csv
"""

import csv
import math
import os
import numpy as np
from common import load_records, CONTINUOUS_COLS, standardise_features


def run_part3():
    rows = load_records()
    n = len(rows)

    X_raw = np.array([[float(r[col]) for col in CONTINUOUS_COLS] for r in rows])
    outcomes = np.array([int(r["churned"]) for r in rows])
    values = np.array([float(r["order_value_inr_hundreds"]) for r in rows])

    # Standardise X1 to X5 over all 800 rows
    X_std, means, sds = standardise_features(X_raw)

    # 3.1 K-means from scratch for K = 2, 3, 4, 5, 6
    wcss_dict = {}
    cluster_labels_dict = {}
    final_centroids_dict = {}

    for K in [2, 3, 4, 5, 6]:
        # Pinned starting centroids
        centroids = np.array([X_std[k * (n // K)] for k in range(K)])
        prev_labels = None
        while True:
            dists = np.sum((X_std[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)  # ties go to lower-numbered centroid
            if prev_labels is not None and np.array_equal(labels, prev_labels):
                break
            prev_labels = labels.copy()
            new_centroids = np.zeros_like(centroids)
            for k in range(K):
                mask = labels == k
                if np.sum(mask) > 0:
                    new_centroids[k] = np.mean(X_std[mask], axis=0)
                else:
                    new_centroids[k] = centroids[k]
            centroids = new_centroids

        # WCSS
        wcss = float(np.sum((X_std - centroids[labels]) ** 2))
        wcss_dict[K] = wcss
        cluster_labels_dict[K] = labels
        final_centroids_dict[K] = centroids

    # 3.2 Choose K with the elbow
    # smallest K in {3, 4, 5} for which WCSS(K) - WCSS(K+1) < 0.5 * (WCSS(K-1) - WCSS(K))
    elbow_k = None
    for K in [3, 4, 5]:
        drop_curr = wcss_dict[K] - wcss_dict[K + 1]
        drop_prev = wcss_dict[K - 1] - wcss_dict[K]
        if drop_curr < 0.5 * drop_prev:
            elbow_k = K
            break

    # 3.3 Profile the K = 3 segments
    labels_3 = cluster_labels_dict[3]
    # order by ascending mean of X3 (discount_order_share, index 2) in original units
    mean_x3_per_cluster = [float(np.mean(X_raw[labels_3 == k, 2])) for k in range(3)]
    cluster_order = np.argsort(mean_x3_per_cluster)  # indices for Segment A, B, C

    os.makedirs("outputs", exist_ok=True)
    seg_names = ["A", "B", "C"]
    seg_sizes = {}
    seg_rows_csv = []
    seg_masks = {}

    for name, c_idx in zip(seg_names, cluster_order):
        mask = labels_3 == c_idx
        seg_masks[name] = mask
        size = int(np.sum(mask))
        seg_sizes[name] = size
        m_x = [float(np.mean(X_raw[mask, j])) for j in range(5)]
        out_rate = float(np.mean(outcomes[mask]))
        m_val = float(np.mean(values[mask]))
        seg_rows_csv.append([
            name,
            size,
            f"{m_x[0]:.4f}",
            f"{m_x[1]:.4f}",
            f"{m_x[2]:.4f}",
            f"{m_x[3]:.4f}",
            f"{m_x[4]:.4f}",
            f"{out_rate:.4f}",
            f"{m_val:.4f}",
        ])

    with open(os.path.join("outputs", "part3_segments.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5", "outcome_rate", "mean_value"])
        writer.writerows(seg_rows_csv)

    # 3.4 Hierarchical clustering on supplied dissimilarity table
    cohorts = ["E1", "E2", "E3", "E4", "E5", "E6"]
    dist_matrix = np.array([
        [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
        [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
        [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
        [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
        [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
        [6.1, 2.0, 1.3, 4.5, 3.4, 0.0],
    ])

    def run_agglomerative(linkage_type):
        clusters = [[i] for i in range(6)]
        merges = []
        for merge_idx in range(1, 6):
            min_dist = float("inf")
            best_pair = None
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    c1, c2 = clusters[i], clusters[j]
                    dists = [dist_matrix[p1][p2] for p1 in c1 for p2 in c2]
                    if linkage_type == "single":
                        d = min(dists)
                    elif linkage_type == "complete":
                        d = max(dists)
                    elif linkage_type == "average":
                        d = sum(dists) / len(dists)
                    if d < min_dist:
                        min_dist = d
                        best_pair = (i, j)

            i, j = best_pair
            c1, c2 = clusters[i], clusters[j]
            name1 = "+".join(sorted([cohorts[idx] for idx in c1]))
            name2 = "+".join(sorted([cohorts[idx] for idx in c2]))

            # cluster_a has smaller minimum member number
            min_num_1 = min(int(cohorts[idx][1:]) for idx in c1)
            min_num_2 = min(int(cohorts[idx][1:]) for idx in c2)
            if min_num_1 < min_num_2:
                cl_a, cl_b = name1, name2
            else:
                cl_a, cl_b = name2, name1

            merges.append((merge_idx, cl_a, cl_b, min_dist))
            new_cl = sorted(c1 + c2)
            clusters = [c for idx, c in enumerate(clusters) if idx not in (i, j)]
            clusters.append(new_cl)

        heights = [m[3] for m in merges]
        gaps = [heights[k] - heights[k - 1] for k in range(1, 5)]
        max_gap_idx = int(np.argmax(gaps))
        merges_below_cut = max_gap_idx + 1
        clusters_at_cut = 6 - merges_below_cut
        return merges, clusters_at_cut

    hierarchical_csv_rows = []
    clusters_at_cut_dict = {}

    for lk in ["single", "complete", "average"]:
        merges, n_cl = run_agglomerative(lk)
        clusters_at_cut_dict[lk] = n_cl
        for m in merges:
            hierarchical_csv_rows.append([lk, m[0], m[1], m[2], f"{m[3]:.4f}"])

    with open(os.path.join("outputs", "part3_hierarchical.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["linkage", "merge_no", "cluster_a", "cluster_b", "height"])
        writer.writerows(hierarchical_csv_rows)

    # 3.5 PCA by the seven steps
    # S = (1/(n-1)) X^T X
    S = (1.0 / (n - 1)) * (X_std.T @ X_std)
    evals, evecs = np.linalg.eigh(S)

    # Sort largest first
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Sign convention: flip each eigenvector so entry of largest absolute value is positive
    for j in range(5):
        col = evecs[:, j]
        max_abs_idx = np.argmax(np.abs(col))
        if col[max_abs_idx] < 0:
            evecs[:, j] = -col

    eigenvalue_sum = float(np.sum(evals))
    exp_ratios = evals / eigenvalue_sum
    cum_ratios = np.cumsum(exp_ratios)
    components_for_80pct = int(np.argmax(cum_ratios >= 0.8) + 1)

    W = evecs[:, :2]
    Z = X_std @ W

    mean_pc1_A = float(np.mean(Z[seg_masks["A"], 0]))
    mean_pc1_B = float(np.mean(Z[seg_masks["B"], 0]))
    mean_pc1_C = float(np.mean(Z[seg_masks["C"], 0]))

    results = [
        ("wcss_k2", f"{wcss_dict[2]:.4f}"),
        ("wcss_k3", f"{wcss_dict[3]:.4f}"),
        ("wcss_k4", f"{wcss_dict[4]:.4f}"),
        ("wcss_k5", f"{wcss_dict[5]:.4f}"),
        ("wcss_k6", f"{wcss_dict[6]:.4f}"),
        ("elbow_k", f"{elbow_k}"),
        ("size_segment_A", f"{seg_sizes['A']}"),
        ("size_segment_B", f"{seg_sizes['B']}"),
        ("size_segment_C", f"{seg_sizes['C']}"),
        ("single_clusters_at_cut", f"{clusters_at_cut_dict['single']}"),
        ("complete_clusters_at_cut", f"{clusters_at_cut_dict['complete']}"),
        ("average_clusters_at_cut", f"{clusters_at_cut_dict['average']}"),
        ("eigenvalue_sum", f"{eigenvalue_sum:.4f}"),
        ("eigenvalue_1", f"{evals[0]:.4f}"),
        ("eigenvalue_2", f"{evals[1]:.4f}"),
        ("eigenvalue_3", f"{evals[2]:.4f}"),
        ("eigenvalue_4", f"{evals[3]:.4f}"),
        ("eigenvalue_5", f"{evals[4]:.4f}"),
        ("explained_ratio_1", f"{exp_ratios[0]:.4f}"),
        ("explained_ratio_2", f"{exp_ratios[1]:.4f}"),
        ("explained_ratio_3", f"{exp_ratios[2]:.4f}"),
        ("explained_ratio_4", f"{exp_ratios[3]:.4f}"),
        ("explained_ratio_5", f"{exp_ratios[4]:.4f}"),
        ("cumulative_ratio_1", f"{cum_ratios[0]:.4f}"),
        ("cumulative_ratio_2", f"{cum_ratios[1]:.4f}"),
        ("cumulative_ratio_3", f"{cum_ratios[2]:.4f}"),
        ("cumulative_ratio_4", f"{cum_ratios[3]:.4f}"),
        ("cumulative_ratio_5", f"{cum_ratios[4]:.4f}"),
        ("components_for_80pct", f"{components_for_80pct}"),
        ("pc1_loading_X1", f"{evecs[0, 0]:.4f}"),
        ("pc1_loading_X2", f"{evecs[1, 0]:.4f}"),
        ("pc1_loading_X3", f"{evecs[2, 0]:.4f}"),
        ("pc1_loading_X4", f"{evecs[3, 0]:.4f}"),
        ("pc1_loading_X5", f"{evecs[4, 0]:.4f}"),
        ("pc2_loading_X1", f"{evecs[0, 1]:.4f}"),
        ("pc2_loading_X2", f"{evecs[1, 1]:.4f}"),
        ("pc2_loading_X3", f"{evecs[2, 1]:.4f}"),
        ("pc2_loading_X4", f"{evecs[3, 1]:.4f}"),
        ("pc2_loading_X5", f"{evecs[4, 1]:.4f}"),
        ("row1_pc1", f"{Z[0, 0]:.4f}"),
        ("row1_pc2", f"{Z[0, 1]:.4f}"),
        ("row2_pc1", f"{Z[1, 0]:.4f}"),
        ("row2_pc2", f"{Z[1, 1]:.4f}"),
        ("row3_pc1", f"{Z[2, 0]:.4f}"),
        ("row3_pc2", f"{Z[2, 1]:.4f}"),
        ("mean_pc1_segment_A", f"{mean_pc1_A:.4f}"),
        ("mean_pc1_segment_B", f"{mean_pc1_B:.4f}"),
        ("mean_pc1_segment_C", f"{mean_pc1_C:.4f}"),
    ]

    with open(os.path.join("outputs", "part3_results.txt"), "w", encoding="utf-8") as f:
        for k, v in results:
            f.write(f"{k} = {v}\n")

    print(f"Part 3 complete: {len(results)} keys written to outputs/part3_results.txt")


if __name__ == "__main__":
    run_part3()

