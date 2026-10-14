# Employee Attrition Prediction — AI-Assisted Data Analytics Final Project

An end-to-end data analytics and machine learning project that predicts **employee attrition** (voluntary turnover) from HR data, and translates the model's findings into concrete business recommendations and a Power BI dashboard design.

**Author:** Nadeen · **Course:** AI-Assisted Data Analytics — Final Project

---

## Project Summary

Employee turnover is one of the most expensive and disruptive problems an organization faces. This project uses the **IBM HR Analytics Employee Attrition & Performance** dataset (1,470 employees, 35 attributes) to answer one question:

> **Which currently-employed staff are at the highest risk of leaving, and which factors drive that risk — so HR can act before they resign, not after?**

The full workflow — data cleaning, exploratory analysis, feature engineering, model training/comparison, and business recommendations — is implemented in a single, heavily documented Jupyter notebook.

### Headline results

| | |
|---|---|
| Overall attrition rate | **16.1%** (237 of 1,470 employees) |
| Strongest behavioral driver | **Overtime** — 30.5% attrition rate vs. 10.4% for non-overtime employees |
| Best model (selected metric: ROC-AUC / Recall) | **Logistic Regression** — ROC-AUC 0.806, Recall 0.660 |
| Other models trained | Decision Tree, Random Forest, Support Vector Machine |
| Top predictive drivers | Income (experience-adjusted), Age, Overtime × Job Level, Total Working Years, Satisfaction |

See `engineeringReport.md` for a complete technical walkthrough of every decision made in the project.

---

## Repository Structure

```
nadeen_final/
├── nadeenFinal.ipynb                  # Main deliverable — the full end-to-end notebook
├── attritionPredictor.html            # Interactive UI to test the trained model in-browser
├── README.md                          # This file
├── engineeringReport.md               # Full technical write-up: every step explained
├── dataset_before/
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv   # Original raw dataset (input, untouched)
│   ├── HR_Attrition_cleaned.csv                # Cleaned dataset (Part 2 output)
│   ├── CleanedFinalData.csv                    # Cleaned dataset — deliverable copy (same content)
│   └── HR_Attrition_feature_engineered.csv     # Cleaned + engineered features (Part 4 output, used for modeling & BI)
```

## Try the Model — `attritionPredictor.html`

A self-contained, single-file interactive UI for testing the selected model (Logistic Regression) on a custom employee profile. **Just double-click the file to open it in any browser** — no server, no build step, no internet connection required.

- Enter a profile (or click **"Load high-risk example"** / **"Load low-risk example"**) across four sections: Demographics, Job Profile, Compensation & Career History, Satisfaction & Performance.
- The 8 engineered features from Part 4 (income-per-experience, promotion-gap ratio, overtime × job-level, etc.) are computed automatically from the raw inputs, exactly as in the notebook.
- Get a live predicted probability, a risk tier (Low/Medium/High), an adjustable decision threshold, and a breakdown of the top factors pushing this specific prediction up or down — mirroring the coefficient interpretability chart in notebook §5.7.
- The model's exact weights (scaler means/scales, one-hot categories, Logistic Regression coefficients) were exported directly from the fitted notebook pipeline, and the in-browser prediction logic was numerically validated against `scikit-learn`'s own `predict_proba` (matches to floating-point precision, <1e-15 difference, across random dataset rows) — the page is a faithful, not an approximate, port of the trained model.
- Visual style matches the notebook's chart palette (light/dark theme aware — use the toggle in the top-right).

## The Notebook — `nadeenFinal.ipynb`

The notebook is organized into 10 parts, each preceded by a markdown cell explaining **what the next code cell does and why**, and followed by a plain-language interpretation of the output:

1. **Dataset Selection & Understanding** — overview, dtypes, missing/duplicate/outlier checks, target variable justification, problem statement
2. **Data Cleaning & Preprocessing** — dropping constants, type correction, outlier capping, saving cleaned data
3. **Exploratory Data Analysis** — 18 charts: histograms, boxplots, count plots, correlation heatmap, scatter, pairplot, violin plot, heatmaps, donut chart, and rate-based trend charts
4. **Feature Engineering** — 8 new business-driven features (e.g. income relative to experience, promotion-gap ratio, overtime × job-level interaction)
5. **Predictive Modeling** — Logistic Regression, Decision Tree, Random Forest, and SVM trained through a leakage-safe Scikit-Learn pipeline
6. **Model Evaluation & Comparison** — Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices, ROC curves, feature importance, coefficient interpretability
7. **AI-Generated Business Recommendations** — executive summary, risk factors, prioritized action list with expected benefit/impact/priority
8. **Power BI Dashboard Design** — 4-page dashboard spec (Executive Overview, Data Exploration, Predictive Analytics, Business Recommendations)
9. **Final Project Report** — full academic write-up (title, abstract, methodology, results, conclusion)
10. **Video Presentation Script** — a ready-to-read 7–10 minute presentation script

## How to Run

**Requirements:** Python 3.10+ with `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, and `jupyter`/`ipykernel`.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

From the project root:

```bash
jupyter notebook nadeenFinal.ipynb
```

Then **Kernel ▸ Restart & Run All**. The notebook is fully self-contained and reproducible (a fixed `RANDOM_STATE = 42` is used throughout) — running it end-to-end regenerates every chart, metric, and the two output CSVs (`HR_Attrition_cleaned.csv` / `CleanedFinalData.csv` and `HR_Attrition_feature_engineered.csv`) from the raw file in `dataset_before/`.

## Data Source

[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) — a widely-used, publicly available benchmark HR dataset (fictional company, realistic structure).

## Key Business Recommendations (see notebook Part 7 for full detail)

1. Cap or better-compensate sustained overtime for junior staff (Job Level 1–2) — the single strongest actionable driver
2. Launch a structured 0–3 year onboarding & mentorship program — attrition is heaviest in early tenure
3. Run a pay-equity review for employees underpaid relative to their experience level
4. Deploy the Logistic Regression model as a quarterly HR "watch-list" to proactively flag at-risk employees
5. Proactively review promotion timelines for employees showing career-stagnation risk
