# Engineering Report — Employee Attrition Prediction Project

This report documents **every technical step and engineering decision** behind `nadeenFinal.ipynb`, in the order the pipeline actually executes. It is written for anyone who needs to understand, audit, reproduce, or extend the project without re-reading 100+ notebook cells — for the plain-language business narrative, see the notebook's Part 9 ("Final Project Report") instead; this document is the technical companion to it.

---

## 1. Scope & Objective

**Goal:** build a reproducible pipeline that takes the raw IBM HR Analytics Employee Attrition CSV and produces (a) a cleaned dataset, (b) a feature-engineered dataset, (c) four trained and evaluated classification models, and (d) business-facing outputs (recommendations, a dashboard spec, a report, a presentation script) — all inside one executable, self-documenting notebook.

**Problem framing:** binary classification. Target = `Attrition` (Yes/No). Business cost is asymmetric — a missed leaver (false negative) is more expensive than an unnecessary retention conversation (false positive) — so the evaluation strategy is built around **Recall** and **ROC-AUC**, not raw Accuracy, from the very first cell of the modeling section onward.

---

## 2. Data Architecture & File Flow

```
dataset_before/WA_Fn-UseC_-HR-Employee-Attrition.csv   (raw input, 1470 rows x 35 cols, never modified)
                │
                ▼  Part 2 — cleaning (drop constants, fix types, cap outliers)
dataset_before/HR_Attrition_cleaned.csv                (1470 rows x 33 cols)
dataset_before/CleanedFinalData.csv                     (identical content — deliverable filename)
                │
                ▼  Part 4 — feature engineering (+10 engineered columns)
dataset_before/HR_Attrition_feature_engineered.csv      (1470 rows x 43 cols, used for modeling + Power BI)
                │
                ▼  Part 5 — in-memory only: train/test split → ColumnTransformer → 4 model pipelines
                    (no intermediate files; everything downstream lives in notebook memory)
```

Two cleaned-data files are written with identical content (`HR_Attrition_cleaned.csv` and `CleanedFinalData.csv`) — the former is the descriptive internal name used throughout the notebook's own logic and cross-references, the latter is the deliverable name requested for external handoff (e.g. attaching to a submission, loading into Power BI without needing to know the notebook's internal naming).

---

## 3. Environment & Reproducibility

| Component | Version used |
|---|---|
| Python | 3.12 |
| pandas | 3.0.3 |
| numpy | 2.5.1 |
| scikit-learn | 1.9.0 |
| matplotlib | 3.11.0 |
| seaborn | 0.13.2 |

**Reproducibility mechanism:** a single constant, `RANDOM_STATE = 42`, is defined once in the setup cell and threaded through every stochastic operation — `train_test_split`, `DecisionTreeClassifier`, `RandomForestClassifier`, `LogisticRegression`, `SVC`. Re-running the notebook top-to-bottom (`Kernel ▸ Restart & Run All`) reproduces byte-identical metrics every time; this was verified by re-executing the notebook twice during development and diffing the results table (see §9 — both runs produced the exact same numbers to 3 decimal places).

The notebook was executed end-to-end via `jupyter nbconvert --to notebook --execute --inplace` and checked programmatically for zero `error`-type cell outputs before being considered complete.

---

## 4. Part 1 Engineering — Data Understanding

- **Column classification:** every one of the 35 raw columns was manually sorted into 4 buckets — continuous numeric (14), ordinal/Likert numeric (9), categorical/nominal (9), identifier-or-constant (3+1) — because each bucket needs different downstream treatment (histograms vs. count plots, scaling vs. one-hot encoding, retention vs. deletion).
- **Missing-value / duplicate check:** `df.isnull().sum()` and `df.duplicated().sum()` both returned 0 — confirmed, not assumed. This was re-verified visually in §3.10 with a nullity heatmap.
- **Outlier detection method:** IQR rule (`Q1 − 1.5×IQR`, `Q3 + 1.5×IQR`) applied to all 14 continuous numeric columns. Seven columns exceeded a meaningful outlier threshold: `YearsSinceLastPromotion`, `MonthlyIncome`, `TotalWorkingYears`, `YearsAtCompany`, `NumCompaniesWorked`, `TrainingTimesLastYear`, `YearsWithCurrManager`.
- **Target selection rationale:** four candidate targets were evaluated (`Attrition`, `MonthlyIncome`, `PerformanceRating`, `JobSatisfaction`). `PerformanceRating` was rejected for having only 2 unique values (near-constant, ~85% at level 3 — effectively unlearnable). `MonthlyIncome`/`JobSatisfaction` were rejected as targets because they are better used as *predictors of* attrition than as prediction targets themselves for this business question. `Attrition` was selected as the only candidate with both strong business relevance and sufficient (if imbalanced) class variation.

---

## 5. Part 2 Engineering — Cleaning Pipeline

Executed as a linear sequence of idempotent operations on a single `df`:

1. **Drop zero-variance identifiers:** `EmployeeCount` (constant=1), `StandardHours` (constant=80), `Over18` (constant='Y'), and `EmployeeNumber` (unique ID, no predictive signal) — verified via `nunique() == 1` before dropping, not assumed from documentation.
2. **Defensive duplicate/missing-value handling:** `drop_duplicates()` and median/mode imputation are applied even though Part 1 found nothing to clean — this makes the pipeline robust to a future, messier export of the same HR system without code changes.
3. **Type correction:** `Attrition_Flag` (int 0/1) and `OverTime_Flag` (int 0/1) are derived from the Yes/No text columns; all remaining text columns are cast to pandas `category` dtype.
4. **Format standardization:** `.str.strip()` on all category columns (defensive; no actual whitespace/casing issues were found).
5. **Outlier treatment — winsorization, not deletion:** the 7 flagged columns are **capped** at their IQR fences rather than having their rows dropped. Rationale: these are legitimate high-tenure/high-income employees, not data errors, and this dataset's positive class (attrition) is already a minority (~16%) — deleting any rows risks discarding rare, informative cases. Capping controls leverage for scale-sensitive models (Logistic Regression, SVM) while preserving every row.
6. **Encoding/scaling deliberately deferred:** neither One-Hot Encoding nor `StandardScaler` is applied in Part 2. This is intentional — applying them before the train/test split (Part 5) would leak test-set statistics (e.g. the scaler's mean/std) into training. They are applied later, inside a `Pipeline`, fit only on `X_train`.

---

## 6. Part 3 Engineering — EDA & Visualization System

**Design system:** a fixed, colorblind-validated 8-color categorical palette plus a 2-color status palette (`#0ca30c` green = retained, `#d03b3b` red = left) is defined once in the setup cell and reused in every one of the 18 charts, so a reader never has to re-learn what a color means between figures. A custom diverging blue→gray→red colormap is used specifically for the correlation heatmap (positive/negative correlation reads as an intuitive hot/cold split around a neutral midpoint). All chart chrome (gridlines, ink colors, surface color) is set once via `plt.rcParams.update(...)`.

**Statistical/visual methods used, and why each was chosen:**

| Method | Where | Why |
|---|---|---|
| Histogram + KDE, split by Attrition | §3.2 | Compares full distribution shape, not just averages |
| Boxplot, split by Attrition | §3.3 | Shows median/IQR/outlier spread compactly; complements the histograms |
| Count plot (raw counts) | §3.4, §3.10 | Shows category composition and volume |
| **Rate**-based bar chart (not count-based) | §3.6, §3.10 | Critical distinction: raw counts are biased by group size (e.g. R&D has more leavers by volume purely because it's the largest department) — dividing by group size gives the metric that actually matters for prioritization |
| Pearson correlation heatmap | §3.5 | Quantifies linear relationships and surfaces multicollinearity among tenure-related fields, which directly informed the Part 4 feature-engineering decisions |
| Scatter plot | §3.7 | Shows how two of the strongest individual predictors (Age, Income) interact jointly |
| Pairplot | §3.8 | Confirms — visually — that no single variable pair cleanly separates the classes, i.e. attrition is a genuinely multivariate pattern, which is the technical justification for using ML models over simple business rules |
| Violin plot (split) | §3.10 | Shows the income-vs-attrition effect holds *within* each job role, not just in aggregate |
| Heatmap (pivot table) | §3.10 | Compact Department × Job Level average-income reference table |
| Donut chart | §3.10 | Communicates the class-imbalance ratio driving the Part 5 modeling strategy |
| Nullity heatmap | §3.10 | Visual (not just numeric) confirmation of the "zero missing values" finding |

---

## 7. Part 4 Engineering — Feature Engineering

Ten new columns are derived from the cleaned data, each targeting a specific pattern surfaced in Part 3 EDA that no single raw column fully captured:

| Feature | Formula | Signal it isolates |
|---|---|---|
| `IncomePerYearWorked` | `MonthlyIncome / (TotalWorkingYears + 1)` | Underpaid-*for-experience*, not just low absolute pay |
| `TenureRatio` | `YearsAtCompany / (TotalWorkingYears + 1)` | Loyalty/mobility — share of career spent at this employer |
| `PromotionGapRatio` | `YearsSinceLastPromotion / (YearsAtCompany + 1)` | Career stagnation relative to tenure |
| `AvgSatisfaction` | mean of 4 Likert satisfaction/engagement columns | Single denoised engagement score |
| `OverTime_x_JobLevel` | `OverTime_Flag * (6 - JobLevel)` | Non-additive interaction: overtime hurts junior staff disproportionately |
| `AgeGroup`, `TenureGroup`, `IncomeBracket` | `pd.cut` / `pd.qcut` bins | Business-readable segments for dashboard/report use (excluded from the ML feature matrix — see §8) |
| `IsJobHopper` | `(NumCompaniesWorked / (TotalWorkingYears+1)) > 0.5` | Binary flag for an established job-mobility pattern |
| `ManagerStability` | `YearsWithCurrManager / (YearsAtCompany + 1)` | Relationship continuity with current manager |

**Validation step (not just construction):** each engineered feature's correlation with `Attrition_Flag` (numeric features) or grouped attrition rate (categorical/flag features) is computed immediately after creation, before being trusted for modeling — several later appear in the top-15 Random Forest importances and Logistic Regression coefficients (§9), confirming they carry real signal rather than being engineering for its own sake.

---

## 8. Part 5 Engineering — Modeling Pipeline

**Feature matrix construction:** `X` excludes `Attrition`, `Attrition_Flag` (target leakage), and the three binned segment columns (`AgeGroup`, `TenureGroup`, `IncomeBracket`) — these are redundant with their continuous source columns already in `X` and are kept only for BI/dashboard readability, not fed to the models.

**Split:** 80/20 `train_test_split` with `stratify=y` — essential given the ~84/16 class imbalance, so both partitions preserve the same attrition rate rather than risking an under-represented positive class in the test set by chance.

**Preprocessing — `ColumnTransformer`:**
- Numeric columns → `StandardScaler` (mean 0, std 1) — necessary because Logistic Regression and SVM are scale-sensitive (Age ranges 18–60, MonthlyIncome ranges into the thousands; without scaling the model would implicitly over-weight large-magnitude features).
- Categorical columns → `OneHotEncoder(handle_unknown="ignore")` — nominal categories have no inherent order, so one-hot is the correct encoding; `handle_unknown="ignore"` makes the pipeline robust to an unseen category at inference time.
- **Fit only on `X_train`**, then applied via `.transform()` to both splits — the single most important leakage-prevention decision in the pipeline. Fitting on the full dataset before splitting (a common mistake) would let test-set means/categories influence the training-time transformation.
- Wrapped inside a per-model `Pipeline([("preprocessor", ...), ("model", ...)])` so preprocessing, training, and inference are always applied consistently and can't drift out of sync.

**Class imbalance handling:** `class_weight="balanced"` is set on every model that supports it (all four). This reweights the loss function inversely proportional to class frequency, so the ~16% minority class isn't drowned out — without it, a model can reach ~84% accuracy by simply always predicting "No", which would be useless for the stated business goal of *finding* leavers.

**Model hyperparameters chosen:**

| Model | Key settings | Rationale |
|---|---|---|
| Logistic Regression | `max_iter=2000`, `class_weight="balanced"` | High `max_iter` ensures convergence after one-hot expansion increases dimensionality |
| Decision Tree | `max_depth=6`, `class_weight="balanced"` | Depth cap is a simple, explicit regularizer against overfitting on a 1,176-row training set |
| Random Forest | `n_estimators=300`, `max_depth=10`, `class_weight="balanced"`, `n_jobs=-1` | 300 trees for stable averaging; depth cap and `n_jobs=-1` for reasonable training time |
| SVM | `kernel="rbf"`, `probability=True`, `class_weight="balanced"` | RBF kernel captures non-linear boundaries; `probability=True` enables ROC-AUC and threshold analysis |

No hyperparameter search (e.g. `GridSearchCV`) was run — hyperparameters were chosen as reasonable, standard defaults appropriate to a dataset of this size, keeping the pipeline fast and the comparison focused on *model family* differences rather than tuning artifacts. This is noted as a limitation in §12.

---

## 9. Part 5–6 Engineering — Evaluation & Model Selection

**Metrics computed for every model:** Accuracy, Precision, Recall, F1, ROC-AUC, full confusion matrix, and ROC curve — deliberately not accuracy alone, because with an 84/16 split, a trivial always-predict-"No" model scores 84% accuracy while being useless.

**Actual results (test set, n=294, 47 positive cases):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.779 | 0.388 | **0.660** | 0.488 | **0.806** |
| Decision Tree | 0.650 | 0.214 | 0.447 | 0.290 | 0.588 |
| Random Forest | 0.827 | 0.455 | 0.426 | 0.440 | 0.790 |
| Support Vector Machine | **0.833** | **0.482** | 0.574 | **0.524** | 0.789 |

**Model-selection engineering decision (documented, not hidden):** the project's first draft assumed Random Forest would be the top performer, based on its typical strength on tabular data. The actual executed results contradicted that: **no model wins on every metric.** SVM has the best F1/Accuracy/Precision; Logistic Regression has the best Recall and ROC-AUC. Because Part 1 committed *in advance* to prioritizing Recall on the positive class (business cost asymmetry: a missed leaver costs more than a false alarm), the model-selection code (`best_model_name = results_df["ROC-AUC"].idxmax()`) and every downstream narrative section were corrected to reflect **Logistic Regression** as the selected model — chosen for a stated, pre-committed reason, not cherry-picked after seeing which model "looks best." This is a deliberate example of letting the evaluation criterion (fixed before results existed) drive the conclusion, rather than the reverse.

**Interpretability, doubled up:** Random Forest's Gini-based feature importances (§5.6) and Logistic Regression's standardized coefficients (§5.7) are both extracted and plotted. They agree on the same top drivers (income, age/experience, overtime), which is used in the report as cross-model corroboration that these drivers reflect real signal rather than one algorithm's idiosyncrasy. The coefficient plot additionally encodes **direction** (color-coded by sign), which a pure importance score cannot show — this is the basis for statements like "overtime increases predicted risk, tenure decreases it."

---

## 10. Business & Presentation Layers (Parts 7–10)

These sections are markdown-only, built by hand-translating the quantitative findings from Parts 3–6 into business language, but two of them are **partially data-driven at the code level**: Part 7 opens with a code cell (`overall_rate`, `overtime_rate`, `young_rate`, `top5_features`, `best_auc`, `best_recall`) that recomputes and prints the key headline numbers live from the executed variables, so the narrative that follows is traceable to actual output rather than hardcoded prose that could drift out of sync after a re-run.

- **Part 7** (Business Recommendations) — 6 prioritized recommendations, each with an explicit Expected Benefit / Business Impact / Priority, derived directly from the EDA (overtime, tenure, income) and modeling (feature importance/coefficients) findings.
- **Part 8** (Power BI Dashboard Design) — a 4-page spec (Executive Overview, Data Exploration, Predictive Analytics, Business Recommendations), each page's KPIs/charts/filters chosen to mirror a specific notebook section, so the dashboard is a direct operationalization of the analysis rather than a disconnected design exercise.
- **Part 9** (Final Report) — a 14-section academic report format, cross-checked against the actual `results_df` numbers (§9 above) after the model-selection correction.
- **Part 10** (Presentation Script) — a timed, spoken-language script referencing the specific charts/cells to show at each timestamp.

---

## 11. File Outputs Reference

| File | Produced by | Rows × Cols | Purpose |
|---|---|---|---|
| `dataset_before/HR_Attrition_cleaned.csv` | Part 2, §2.8 | 1470 × 33 | Cleaned data, internal notebook reference name |
| `dataset_before/CleanedFinalData.csv` | Part 2, §2.8 | 1470 × 33 | Cleaned data, deliverable filename (identical content) |
| `dataset_before/HR_Attrition_feature_engineered.csv` | Part 4 | 1470 × 43 | Cleaned + engineered features, used for modeling and recommended as the Power BI data source |

---

## 12. Known Limitations & Suggested Future Work

- **No hyperparameter search.** All four models use reasonable defaults rather than tuned parameters (e.g. `GridSearchCV`/`RandomizedSearchCV`). A tuned Random Forest or SVM could plausibly close the Recall gap with Logistic Regression — this is the highest-value next step if the project were extended.
- **Single train/test split, no cross-validation.** With only 294 test rows and 47 positive cases, metric estimates carry meaningful sampling variance; k-fold cross-validation would give more stable estimates and confidence intervals.
- **No SHAP/permutation-importance analysis.** Random Forest's Gini importance and Logistic Regression's coefficients are informative but have known biases (Gini importance favors high-cardinality features; coefficients assume linearity/no strong interaction effects beyond the one explicitly engineered). SHAP values would give a more rigorous, model-agnostic attribution.
- **Static dataset, single time snapshot.** The data has no dates/time series component, so the model predicts attrition risk as a cross-sectional pattern, not a time-to-event (survival analysis) — a Cox proportional hazards or time-to-attrition model would be a natural extension if longitudinal data became available.
- **Threshold left at the default 0.5.** Given the stated Recall priority, an explicit threshold-tuning step (e.g. picking the threshold that maximizes F2 score, or that hits a target Recall) is a documented but unimplemented option (mentioned in Part 6.2) — implementing it would let the business directly trade off Recall vs. Precision without retraining.

---

## 13. How to Reproduce This Report

1. Ensure the raw file `dataset_before/WA_Fn-UseC_-HR-Employee-Attrition.csv` is present.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn jupyter`.
3. Run `jupyter nbconvert --to notebook --execute --inplace nadeenFinal.ipynb` (or open in Jupyter and `Restart & Run All`).
4. Confirm zero error outputs and that `dataset_before/CleanedFinalData.csv` and `dataset_before/HR_Attrition_feature_engineered.csv` were regenerated.

`RANDOM_STATE = 42` guarantees the metrics table in §9 will reproduce exactly.
