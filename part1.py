"""Part 1 — Estimation, inference and a regression baseline.
Produces outputs/part1_results.txt.
"""

import math
import os
import numpy as np
from common import load_records


def run_part1():
    rows = load_records()
    n = len(rows)

    values = np.array([float(r["order_value_inr_hundreds"]) for r in rows])
    x1 = np.array([float(r["customer_tenure_months"]) for r in rows])
    x2 = np.array([float(r["complaint_count_90d"]) for r in rows])
    x3 = np.array([float(r["discount_order_share"]) for r in rows])
    x4 = np.array([float(r["avg_order_value_inr_hundreds"]) for r in rows])
    x5 = np.array([float(r["loyalty_score"]) for r in rows])
    x6 = np.array([float(r["cod_flag"]) for r in rows])
    x7 = np.array([float(r["plus_member_flag"]) for r in rows])

    # 1.1 Estimate with an interval
    mean_value = np.mean(values)
    sd_value = np.std(values, ddof=1)
    se_value = sd_value / math.sqrt(n)
    ci95_low = mean_value - 1.96 * se_value
    ci95_high = mean_value + 1.96 * se_value

    # 1.2 Test a planning claim
    z_stat = (mean_value - 8.0) / se_value
    reject_h0 = "yes" if abs(z_stat) > 1.96 else "no"

    # 1.3 Size a future sample
    E = 0.5
    n_required = math.ceil((1.96 * sd_value / E) ** 2)

    # 1.4 Simple regression by closed form
    mean_x4 = np.mean(x4)
    cov_x4_val = np.sum((x4 - mean_x4) * (values - mean_value)) / (n - 1)
    var_x4 = np.sum((x4 - mean_x4) ** 2) / (n - 1)
    reg_slope = cov_x4_val / var_x4
    reg_intercept = mean_value - reg_slope * mean_x4
    fitted = reg_intercept + reg_slope * x4

    tss = np.sum((values - mean_value) ** 2)
    ess = np.sum((fitted - mean_value) ** 2)
    rss = np.sum((values - fitted) ** 2)
    tss_equals_ess_plus_rss = "yes" if abs(tss - ess - rss) < 1e-6 * tss else "no"
    r2 = 1.0 - (rss / tss)
    rmse = math.sqrt(rss / n)
    mae = float(np.mean(np.abs(values - fitted)))

    # 1.5 Fit the same line by gradient descent
    sd_x4 = np.std(x4, ddof=1)
    z = (x4 - mean_x4) / sd_x4
    m = 0.0
    c = 0.0
    eta = 0.1

    def compute_cost(m_val, c_val):
        err = values - (m_val * z + c_val)
        return float(np.mean(err ** 2))

    gd_J0 = compute_cost(m, c)
    gd_J1 = None
    gd_J10 = None
    gd_J100 = None
    gd_J500 = None

    for step in range(1, 501):
        err = values - (m * z + c)
        dm = -(2.0 / n) * np.sum(z * err)
        dc = -(2.0 / n) * np.sum(err)
        m -= eta * dm
        c -= eta * dc
        if step == 1:
            gd_J1 = compute_cost(m, c)
        elif step == 10:
            gd_J10 = compute_cost(m, c)
        elif step == 100:
            gd_J100 = compute_cost(m, c)
        elif step == 500:
            gd_J500 = compute_cost(m, c)

    gd_slope = m / sd_x4
    gd_intercept = c - m * (mean_x4 / sd_x4)
    gd_slope_gap = abs(gd_slope - reg_slope)

    # 1.6 Check for a funnel, then apply the log remedy
    sort_order_raw = np.argsort(fitted, kind="stable")
    residuals_raw = values - fitted
    resid_lower_raw = residuals_raw[sort_order_raw[:400]]
    resid_upper_raw = residuals_raw[sort_order_raw[400:]]
    resid_sd_lower_raw = np.std(resid_lower_raw, ddof=1)
    resid_sd_upper_raw = np.std(resid_upper_raw, ddof=1)
    resid_sd_ratio_raw = resid_sd_upper_raw / resid_sd_lower_raw

    ln_val = np.log(values)
    ln_x4 = np.log(x4)
    mean_ln_x4 = np.mean(ln_x4)
    mean_ln_val = np.mean(ln_val)

    cov_log = np.sum((ln_x4 - mean_ln_x4) * (ln_val - mean_ln_val)) / (n - 1)
    var_log = np.sum((ln_x4 - mean_ln_x4) ** 2) / (n - 1)
    loglog_slope = cov_log / var_log
    loglog_intercept = mean_ln_val - loglog_slope * mean_ln_x4
    fitted_log = loglog_intercept + loglog_slope * ln_x4

    tss_log = np.sum((ln_val - mean_ln_val) ** 2)
    rss_log = np.sum((ln_val - fitted_log) ** 2)
    loglog_r2 = 1.0 - (rss_log / tss_log)
    loglog_rmse = math.sqrt(rss_log / n)

    sort_order_log = np.argsort(fitted_log, kind="stable")
    residuals_log = ln_val - fitted_log
    resid_lower_log = residuals_log[sort_order_log[:400]]
    resid_upper_log = residuals_log[sort_order_log[400:]]
    resid_sd_lower_log = np.std(resid_lower_log, ddof=1)
    resid_sd_upper_log = np.std(resid_upper_log, ddof=1)
    resid_sd_ratio_log = resid_sd_upper_log / resid_sd_lower_log

    # 1.7 Multiple regression by the normal equation
    X_mat = np.column_stack([np.ones(n), ln_x4, x1, x2, x3, x5, x6, x7])
    Y_vec = ln_val
    beta = np.linalg.solve(X_mat.T @ X_mat, X_mat.T @ Y_vec)

    fitted_mr = X_mat @ beta
    rss_mr = np.sum((Y_vec - fitted_mr) ** 2)
    tss_mr = np.sum((Y_vec - np.mean(Y_vec)) ** 2)
    mr_r2 = 1.0 - (rss_mr / tss_mr)
    p = 7
    mr_adj_r2 = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - p - 1)
    loglog_adj_r2 = 1.0 - (1.0 - loglog_r2) * (n - 1) / (n - 1 - 1)

    os.makedirs("outputs", exist_ok=True)
    results = [
        ("n", f"{n}"),
        ("mean_value", f"{mean_value:.4f}"),
        ("sd_value", f"{sd_value:.4f}"),
        ("se_value", f"{se_value:.4f}"),
        ("ci95_low", f"{ci95_low:.4f}"),
        ("ci95_high", f"{ci95_high:.4f}"),
        ("z_stat", f"{z_stat:.4f}"),
        ("reject_h0", f"{reject_h0}"),
        ("n_required", f"{n_required}"),
        ("reg_slope", f"{reg_slope:.4f}"),
        ("reg_intercept", f"{reg_intercept:.4f}"),
        ("tss", f"{tss:.4f}"),
        ("ess", f"{ess:.4f}"),
        ("rss", f"{rss:.4f}"),
        ("tss_equals_ess_plus_rss", f"{tss_equals_ess_plus_rss}"),
        ("r2", f"{r2:.4f}"),
        ("rmse", f"{rmse:.4f}"),
        ("mae", f"{mae:.4f}"),
        ("gd_J0", f"{gd_J0:.4f}"),
        ("gd_J1", f"{gd_J1:.4f}"),
        ("gd_J10", f"{gd_J10:.4f}"),
        ("gd_J100", f"{gd_J100:.4f}"),
        ("gd_J500", f"{gd_J500:.4f}"),
        ("gd_slope", f"{gd_slope:.4f}"),
        ("gd_intercept", f"{gd_intercept:.4f}"),
        ("gd_slope_gap", f"{gd_slope_gap:.4f}"),
        ("resid_sd_lower_raw", f"{resid_sd_lower_raw:.4f}"),
        ("resid_sd_upper_raw", f"{resid_sd_upper_raw:.4f}"),
        ("resid_sd_ratio_raw", f"{resid_sd_ratio_raw:.4f}"),
        ("loglog_intercept", f"{loglog_intercept:.4f}"),
        ("loglog_slope", f"{loglog_slope:.4f}"),
        ("loglog_r2", f"{loglog_r2:.4f}"),
        ("loglog_rmse", f"{loglog_rmse:.4f}"),
        ("resid_sd_lower_log", f"{resid_sd_lower_log:.4f}"),
        ("resid_sd_upper_log", f"{resid_sd_upper_log:.4f}"),
        ("resid_sd_ratio_log", f"{resid_sd_ratio_log:.4f}"),
        ("mr_beta_0", f"{beta[0]:.4f}"),
        ("mr_beta_1", f"{beta[1]:.4f}"),
        ("mr_beta_2", f"{beta[2]:.4f}"),
        ("mr_beta_3", f"{beta[3]:.4f}"),
        ("mr_beta_4", f"{beta[4]:.4f}"),
        ("mr_beta_5", f"{beta[5]:.4f}"),
        ("mr_beta_6", f"{beta[6]:.4f}"),
        ("mr_beta_7", f"{beta[7]:.4f}"),
        ("mr_r2", f"{mr_r2:.4f}"),
        ("mr_adj_r2", f"{mr_adj_r2:.4f}"),
        ("loglog_adj_r2", f"{loglog_adj_r2:.4f}"),
    ]

    with open(os.path.join("outputs", "part1_results.txt"), "w", encoding="utf-8") as f:
        for k, v in results:
            f.write(f"{k} = {v}\n")

    print(f"Part 1 complete: {len(results)} keys written to outputs/part1_results.txt")


if __name__ == "__main__":
    run_part1()

