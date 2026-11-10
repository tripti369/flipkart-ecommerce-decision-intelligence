Domain: E-commerce

## How to run

To replicate all analyses and regenerate every results file from a clean clone, run the following commands from the repository root:

```bash
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

All output summaries and tabular CSV files will be generated in the `outputs/` directory.

---

## Part 1 findings

### R1.1 Hypothesis Test on Average Order Value
- The null hypothesis H0 (μ = 8.0 hundred INR, i.e., INR 800) is **rejected** at significance level α = 0.05.
- The calculated test statistic is `z_stat = 2.8930`, which exceeds the two-tailed critical threshold of 1.96 (`reject_h0 = yes`).
- **Type I error definition:** In this domain context, a Type I error would occur if the analytics team concluded that true average customer order value has changed from INR 800 when in reality the true population mean is still exactly INR 800, leading to unwarranted budget reallocations across category inventory plans.
- **Practical significance:** The sample mean is `mean_value = 8.9382` (representing INR 893.82), yielding an observed gap of +0.9382 column units (INR 93.82 per order, an 11.7% increase over the planning baseline). In large-scale e-commerce operations with millions of monthly transactions, an incremental ~INR 94 per order represents substantial top-line revenue and significantly impacts working capital, warehouse logistics, and category gross merchandise value (GMV) targets; therefore, this difference is practically meaningful.

### R1.2 Sample Size Sizing
- Sizing a future sample to guarantee a margin of error within E = 0.5 column units (INR 50) at 95% confidence requires `n_required = 1293` observations.
- Comparing `n_required = 1293` to the currently available 800 records demonstrates that the existing dataset is insufficient to achieve the planning team's desired INR 50 precision target. Category planners cannot treat current estimates as tightly bounded to within INR 50 without gathering at least 493 additional transaction records.

### R1.3 Gradient Descent vs. Closed-Form Ordinary Least Squares
- The gradient descent line matches the closed-form OLS analytical line with `gd_slope_gap = 0.0000` (both yield slope 0.6701 and intercept -3.6598). After 500 iterations, gradient descent produced a slope close to the closed-form regression slope, with the difference falling within the required numerical tolerance.

### R1.4 Funnel Effect and Logarithmic Transformation
- The raw linear model exhibits severe heteroskedasticity with `resid_sd_lower_raw = 2.3155`, `resid_sd_upper_raw = 8.6891`, and a ratio of `resid_sd_ratio_raw = 3.7525`, confirming a pronounced funnel where residual dispersion expands dramatically at higher predicted order values.
- In contrast, the log transformation substantially reduced the observed heteroscedasticity pattern, as indicated by the residual standard deviation ratio moving closer to 1 (`resid_sd_lower_log = 0.5165`, `resid_sd_upper_log = 0.5369`, and `resid_sd_ratio_log = 1.0396`).

### R1.5 Multiple Regression Coefficients and Model Comparison
- In Task 1.7, the predictors whose coefficients have an absolute magnitude above 0.05 (excluding the intercept `mr_beta_0 = -1.8613`) are:
  - `ln(X4)` (`avg_order_value_inr_hundreds` in logs): `mr_beta_1 = 1.1576`
  - `X2` (`complaint_count_90d`): `mr_beta_3 = 0.0940`
  - `X3` (`discount_order_share`): `mr_beta_4 = 0.1067`
  - `X6` (`cod_flag`): `mr_beta_6 = 0.1518`
- The multiple regression model achieved an adjusted R-squared of 0.7403 compared with 0.6561 for the log-log regression model. This indicates a better model fit after accounting for the number of predictors, although it does not independently establish statistical significance.

---

## Part 2 findings

### R2.1 Learning Rate Behavior in Logistic Regression
- Ordering the training losses after 200 epochs from lowest to highest:
  1. η = 1.0: `loss_epoch_200_eta_1.0 = 0.3703`
  2. η = 0.1: `loss_epoch_200_eta_0.1 = 0.3776`
  3. η = 0.01: `loss_epoch_200_eta_0.01 = 0.5091`
- The smaller learning rate (0.01) progresses more slowly within 200 epochs, while the learning rate of 1.0 achieves the lowest training loss among the tested values. This comparison is based on the reported loss after 200 epochs.

### R2.2 Key Churn Drivers and Odds Ratio Interpretation
- The three predictors with the largest absolute standardized weights are:
  1. `X3` (`discount_order_share`): `weight_X3 = 0.9013` (odds ratio = 2.4629)
  2. `X1` (`customer_tenure_months`): `weight_X1 = -0.5554` (odds ratio = 0.5739)
  3. `X2` (`complaint_count_90d`): `weight_X2 = 0.5505` (odds ratio = 1.7341)
- **Domain interpretation of largest odds ratio:** For every one standard deviation increase in a customer's discount order share (~0.23), the odds of the customer churning within the next 90 days are multiplied by 2.4629 (a 146.3% increase in churn odds), holding all other predictors constant.

### R2.3 Evaluation at Multiple Thresholds vs. Blind Baseline
- The blind baseline accuracy (predicting 0 for all records) is `blind_accuracy = 0.7875` (78.75%), which is barely below the model's accuracy at τ = 0.5 (`tau_0.5_accuracy = 0.8042` or 80.42%). Accuracy alone is a misleading metric because an uninformative model that flags zero churners scores nearly 79% simply by exploiting class imbalance (51 positives out of 240 test records).
- Lowering the decision threshold τ from 0.5 to 0.3 increased recall from `0.4706` (24/51 churners caught) to `0.7451` (38/51 caught), while precision decreased from `0.5455` to `0.4634` due to false positives increasing from 20 to 44.

### R2.4 Logistic Regression vs. K-Nearest Neighbours
- Comparing test set performance:
  - **Logistic regression (τ = 0.3):** `f1 = 0.5714`, `recall = 0.7451` (precision = 0.4634)
  - **KNN (K = 5):** `knn_f1 = 0.3696`, `knn_recall = 0.3333` (precision = 0.4146)
- Logistic regression substantially outperforms KNN, capturing over twice the proportion of churners (74.5% vs. 33.3%) and delivering a +0.2018 higher F1 score.

### R2.5 Financial Cost Optimization and Threshold Selection
- Evaluating asymmetric retention economics (false positive coupon cost = INR 50; false negative lost margin = INR 400) identifies an optimal training threshold `cost_tau = 0.1500` with training cost `cost_train_at_tau = 12150` (INR 12,150).
- Applying this threshold to the test set yields:
  - Optimal threshold test cost (`cost_test_at_tau`): **INR 6,050** (38 TP, 43 FP, 8 FN)
  - Default threshold test cost (`cost_test_at_0.5`): **INR 11,800** (24 TP, 20 FP, 27 FN)
  - Unflagged baseline cost (`cost_test_flag_nobody`): **INR 20,400** (0 TP, 0 FP, 51 FN)
- **Deployment recommendation:** Deploy the logistic model at τ = 0.1500, which reduces business cost by 48.7% compared to τ = 0.5 and saves INR 14,350 relative to inaction.
- The threshold was determined exclusively on training rows to prevent data leakage and avoid overfitting decision policy parameters to sample noise in the test split.

---

## Part 3 findings

### R3.1 Elbow Analysis and Cluster Selection
- Successive drops in within-cluster sum of squares (WCSS) across candidate K values:
  - WCSS(2) − WCSS(3) = 2262.6753 − 1602.8471 = **659.8282**
  - WCSS(3) − WCSS(4) = 1602.8471 − 1494.7732 = **108.0739**
  - WCSS(4) − WCSS(5) = 1494.7732 − 1323.7157 = **171.0575**
  - WCSS(5) − WCSS(6) = 1323.7157 − 1278.1691 = **45.5466**
- The elbow criterion identifies `elbow_k = 3` because the drop from K=3 to K=4 (108.0739) is less than half the preceding drop from K=2 to K=3 (0.5 × 659.8282 = 329.9141). Although the subsequent drops do not shrink steadily (since pinned deterministic seeds yield a reproducible local solution rather than a guaranteed global optimum), based on the specified elbow criterion, K=3 is selected for customer segmentation. The resulting clusters are then interpreted using their customer profiles and churn rates.

### R3.2 Customer Segment Profiles and Actions
From `outputs/part3_segments.csv`:
1. **Segment A (Size 322) — "Core Loyalists"**:
   - *Profile:* Longest tenure (mean 47.5 months), low friction (mean 1.95 complaints), very low discount reliance (mean discount share 0.2250), high loyalty score (mean 62.16), and modest average order value (mean INR 359.76).
   - *Action:* Reward with Flipkart Plus tier status, early sale access, and loyalty milestone rewards rather than price cuts.
2. **Segment B (Size 255) — "High-Value Frustrated Shoppers"**:
   - *Profile:* Intermediate tenure (mean 28.6 months), severe service friction (mean 6.07 complaints), moderate discount share (0.4534), highest average order value (mean INR 1,712.48), and solid loyalty score (mean 55.01).
   - *Action:* Route complaints to dedicated white-glove customer support and prioritize fulfillment dispatch to safeguard high-margin transactions.
3. **Segment C (Size 223) — "Bargain-Focused Churn Risks"**:
   - *Profile:* Short tenure (mean 14.4 months), elevated friction (mean 4.23 complaints), highest discount reliance (mean discount share 0.7118), low loyalty score (mean 38.52), and intermediate order value (mean INR 728.83).
   - *Highest outcome rate:* Segment C has the highest churn rate at `outcome_rate = 0.4664` (46.64%).
   - *Action:* Deploy targeted retention coupons and flash sales with minimum cart thresholds to re-engage accounts before their 90-day lapse window expires.

### R3.3 Hierarchical Clustering Cut Discrepancies
- **Clusters at dendrogram cut:**
  - *Single linkage* (`single_clusters_at_cut = 4`): `{E2, E3, E6}`, `{E1}`, `{E4}`, `{E5}`.
  - *Complete linkage* (`complete_clusters_at_cut = 2`): `{E1, E4, E5}` and `{E2, E3, E6}`.
- Single linkage measures minimum pairwise distance between groups, causing it to chain mutually close cohorts (E2, E3, E6) early while leaving more dispersed individual cohorts unmerged across a large height gap. In contrast, complete linkage evaluates maximum pairwise distance, resisting chaining and enforcing compact clusters that merge E1, E4, and E5 together before encountering a large final merge height (7.9000).

### R3.4 Principal Component Analysis
- **First component drivers:** PC1 accounts for 52.27% of total variance and is driven positively by tenure X1 (+0.5317) and loyalty score X5 (+0.4418), and negatively by discount share X3 (-0.5187) and complaint count X2 (-0.4100). It serves as an index of customer account maturity and relationship health versus operational friction and discount sensitivity.
- **Variance threshold:** 3 components are required to capture at least 80% of total variance (`components_for_80pct = 3`, cumulative variance = 86.16%).
- **Segment separation:** Mean PC1 scores by segment are `mean_pc1_segment_A = 1.7773`, `mean_pc1_segment_B = -0.7831`, and `mean_pc1_segment_C = -1.6709`. This demonstrates that PC1 aligns with customer segment structure, providing clean separation between loyal, stable accounts (positive scores) and vulnerable, deal-seeking accounts (negative scores).

---

## Part 4 findings

### R4.1 Split Purity and Overfitting Control
- Split S4 (`customer_tenure_months <= 24`) is the best split because it achieves the highest information gain (0.1252 bits) and Gini gain (0.0606), isolating new accounts that contain a heavy concentration of churners (86 positives out of 188 records).
- Growing a decision tree until every leaf is pure causes the model to memorize noise, idiosyncrasies, and outliers in the training sample, creating an over-parameterized model that suffers from high variance and poor generalization.
- Setting a maximum tree depth constraint (such as `max_depth = 3`) or requiring a minimum number of samples per leaf (such as `min_samples_leaf = 20`) effectively prevents over-branching and controls overfitting.

### R4.2 Model Selection from Voting Ensemble
- Comparing test performance across the five candidate models:
  - **Logistic regression:** `f1 = 0.5714`, `recall = 0.7451` (precision = 0.4634)
  - **KNN (K = 5):** `f1 = 0.3696`, `recall = 0.3333` (precision = 0.4146)
  - **Decision stump (S4):** `f1 = 0.5532`, `recall = 0.7647` (precision = 0.4333)
  - **Hard vote ensemble:** `f1 = 0.5344`, `recall = 0.6863` (precision = 0.4375)
  - **Soft vote ensemble:** `f1 = 0.5507`, `recall = 0.7451` (precision = 0.4368)
- I recommend deploying the standalone **logistic regression** model. Combining models did not improve performance; both voting ensembles underperformed logistic regression on F1 because ensemble decisions were diluted by the weak predictive power of KNN.

### R4.3 AdaBoost Weights and Error Updates
- The total normalized weight of the two misclassified training points (points 3 and 7) after Round 1 normalization is exactly **0.5000** (0.2500 each).
- Round 2 places greater confidence in h2 than h1 (trust weight α2 = 0.8047 vs. α1 = 0.5493) because h2 achieves a lower weighted error rate (ε2 = 0.1667 vs. ε1 = 0.2500) on the re-weighted sample distribution by correctly classifying the upweighted points.

### R4.4 Neural Network Sizing and Deep Learning Mechanics
- **Model A (Option 2 CNN) Layer Architecture:**

| Layer | Output Shape (H × W × D) | Parameter Count |
|---|---|---|
| `cnn_conv1` (5×5 conv, 6 filters, pad 2, stride 1) | 28 × 28 × 6 | 156 |
| `cnn_pool1` (2×2 max pool, stride 2) | 14 × 14 × 6 | 0 |
| `cnn_conv2` (5×5 conv, 16 filters, pad 0, stride 1) | 10 × 10 × 16 | 2,416 |
| `cnn_pool2` (2×2 max pool, stride 2) | 5 × 5 × 16 | 0 |
| `cnn_dense1` (flatten 400 → 32 dense) | 32 | 12,832 |
| `cnn_dense2` (32 → 10 dense output) | 10 | 330 |
| **Total CNN Parameters (`cnn_total`)** | — | **15,734** |

- **Model B Recurrent Neural Network:**
  - Weight matrix dimensions: `W_xh` is (128, 64), `W_hh` is (128, 128), `b_h` is (128,), `W_hy` is (16, 128), and `b_y` is (16,).
  - The parameter count `rnn_total = 26768` does not scale with sequence length because the recurrent cell shares the exact same weight matrices across all sequence time steps.
- **Activations and Loss for 10-Class Classification:**
  - Hidden-layer activation: **ReLU** (Rectified Linear Unit)
  - Output activation: **Softmax**
  - Loss function: **Categorical Cross-Entropy**
- **Vanishing Gradients and Residual Connections:**
  - In deep feedforward networks, repeated multiplications of fractional derivatives during backpropagation cause gradient signals to exponentially attenuate toward zero in earlier layers, halting parameter learning. Skip (residual) connections provide an identity shortcut ($H(x) = F(x) + x$) that permits gradients to flow backward undiminished across layers, preserving gradient magnitude throughout deep architectures.

---

## Model-choice memo

**To:** Head of Retention Operations, Flipkart  
**From:** Customer Analytics Decision Intelligence Team  
**Subject:** Production Churn Alert Model Recommendation  

We recommend deploying **logistic regression trained using gradient descent** at decision threshold **τ = 0.1500** for production retention interventions.

In evaluating customer churn risk, asymmetric operational costs dominate: issuing a preemptive retention coupon to an active shopper costs INR 50, whereas allowing an at-risk customer to churn unflagged incurs an estimated INR 400 in lost future operating margin. Using the test set of 240 customer accounts, deploying logistic regression at the financially optimal threshold of τ = 0.1500 generates a total intervention cost of **INR 6,050** (`cost_test_at_tau = 6050`). This represents a 48.7% cost reduction compared to the conventional τ = 0.5000 threshold cost of **INR 11,800** (`cost_test_at_0.5 = 11800`), and saves **INR 14,350** relative to the unflagged baseline cost of **INR 20,400** (`cost_test_flag_nobody = 20400`).

Among tested alternatives, logistic regression achieved the highest F1 score among the tested models (0.5714), while maintaining a recall of 0.7451 at τ = 0.3, compared to K-nearest neighbours (`knn_f1 = 0.3696`, `knn_recall = 0.3333`) and voting ensembles (`hard_vote f1 = 0.5344`). Combining models diluted performance because neighbor-based voting introduced substantial classification noise.

**Operational Risk:** Our evidence is subject to key limitations: the analysis relies on a synthetic dataset and a single 70/30 train-test split containing only 51 positive churn events; slight shifts in real customer behavior or coupon response decay could alter false-alarm economics.

Neither Model A (product image sorting) nor Model B (review text classification) is worth building before the tabular retention model is operational in production, as tabular churn alerts address immediate margin loss with established operational levers.

---

## Limitations

1. **Synthetic Data Realism:** All customer records were generated via synthetic parametric distributions. While statistical correlations are controlled, synthetic distributions do not account for macro-economic retail trends, festival sale spikes (e.g., Big Billion Days), regional delivery disruptions, or category-specific supply bottlenecks.
2. **Single Stratified Split Sensitivity:** All validation results rely on a single 70/30 train/test split. Although stratified by churn outcome, performance metrics remain susceptible to split-specific sampling variation; repeated k-fold cross-validation across all models would provide narrower empirical confidence intervals.
3. **Small Positive Sample Size:** The test set contains only 51 positive churn records. At this sample size, a classification difference of merely two customer accounts shifts sensitivity or precision by approximately 4 percentage points, inflating metric volatility.
4. **Static Cost Assumptions:** The financial objective function fixes misclassification costs at INR 50 per false alarm and INR 400 per missed churner. In actual operations, lost margins scale with customer lifetime value and order volume, meaning high-value customers justify significantly higher retention spend than low-basket deal seekers.
