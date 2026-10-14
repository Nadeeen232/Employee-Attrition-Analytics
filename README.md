# Employee Attrition Analytics

An end-to-end data analytics, machine learning, and business intelligence project that investigates the factors associated with voluntary employee turnover and translates the findings into actionable retention strategies.

The project analyzes the **IBM HR Analytics Employee Attrition & Performance** dataset using Python, machine learning, and a three-page Power BI dashboard design.

## Business Problem

Employee attrition can increase recruitment costs, disrupt operations, and reduce organizational knowledge. This project aims to help HR teams answer the following questions:

- Which employee groups have the highest attrition rates?
- What factors are most strongly associated with employee turnover?
- Can machine learning identify employees who may be at higher risk of leaving?
- Which retention actions should the organization prioritize?

## Dataset

- **Dataset:** IBM HR Analytics Employee Attrition & Performance
- **Source:** Kaggle
- **Original records:** 1,470 employees
- **Original variables:** 35
- **Cleaned records:** 1,470 unique employees
- **Final columns:** 47, including 10 engineered features
- **Target variable:** `Attrition` (`Yes` / `No`)

The variables cover:

- Employee demographics
- Compensation and benefits
- Job role and department
- Tenure and professional experience
- Satisfaction and work-life balance
- Business travel and overtime
- Performance-related information

> Download the original dataset from Kaggle unless its license explicitly permits redistribution.

## Project Workflow

1. Business and dataset understanding
2. Data-quality assessment
3. Data cleaning and validation
4. Feature engineering
5. Exploratory data analysis
6. KPI definition
7. Machine-learning model development
8. Model evaluation and selection
9. Business insight generation
10. Power BI dashboard design
11. Retention recommendations

## Data Cleaning

A documented Python pipeline using pandas and NumPy addressed six categories of data-quality issues in the working dataset:

- Imputed 40 missing values using group-wise medians or categorical modes
- Removed 15 exact duplicate employee records using `EmployeeNumber`
- Standardized 21 inconsistent categorical labels
- Converted 10 incorrectly typed numerical values back to numeric form
- Detected and corrected 8 unrealistic outliers using business-rule boundaries
- Corrected 2 scale-corruption errors using a rescaling rule

The source Kaggle dataset was originally clean. Realistic data-quality issues were deliberately introduced into a working copy to demonstrate a complete cleaning and validation workflow.

## Feature Engineering

Ten analytical features were created, including:

- `SatisfactionIndex`
- `IncomeBand`
- `TenureRatio`
- Tenure-related segments
- Employee risk and grouping variables used in the analysis

## Exploratory Data Analysis

The analysis includes:

- Summary statistics
- Missing-value validation
- Distribution histograms
- Income box plots
- Correlation analysis
- Categorical segment comparisons
- Tenure trend analysis
- Grouped aggregations
- Attrition-class analysis

## Key Findings

- The overall employee attrition rate was **16.1%**.
- Employees working overtime had an attrition rate of **30.5%**, compared with **10.4%** for employees not working overtime.
- Employees within their first two years at the organization had an attrition rate of **29.8%**.
- Sales Representatives recorded the highest role-level attrition rate at **39.8%**.
- Single employees had an attrition rate of **25.6%**, compared with **12.4%** for married employees.
- Employees under 30 had an attrition rate of **27.8%**, compared with **9.8%** for employees aged 40 to 49.
- Attrition decreased from **29.8%** during the first two years to **8.1%** after 11 years.
- Overtime showed the strongest measured correlation with attrition at approximately **0.245**.
- Employees who left had lower pay than employees who stayed across every department.
- The largest department-level pay gap was observed in Human Resources at approximately **3,461 per month**.

These results describe patterns and associations in the available data and should not be interpreted as proof of causation.

## Machine Learning

Three supervised classification models were trained using a stratified 80/20 train-test split:

- Logistic Regression
- Decision Tree
- Random Forest

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 87.1% | 73.7% | 29.8% | 0.424 | 0.777 |
| Decision Tree | 83.7% | Not reported | Not reported | 0.351 | 0.694 |
| Random Forest | 85.4% | Not reported | 12.8% | Not reported | 0.792 |

## Selected Model

**Logistic Regression** was selected as the recommended model because it provided the best reported precision-recall balance, achieved the highest F1-score, and remained interpretable for HR stakeholders.

All three models ranked overtime as a top predictor, supporting the findings from the exploratory analysis.

> The model should be used as a decision-support tool rather than as an automated system for making employment decisions.

## KPI Analysis

| KPI | Actual | Target | Status |
|---|---:|---:|---|
| Overall Attrition Rate | 16.1% | 12% | Below target |
| Overtime Attrition Rate | 30.5% | 20% | Below target |
| Early-Tenure Attrition Rate | 29.8% | 20% | Below target |
| Retention Model Precision | 73.7% | 70% | On/above target |

Three of the four KPIs require improvement, while model precision exceeds its operational target.

## Power BI Dashboard

The dashboard is designed across three pages:

1. **Executive Summary**
   - Headline KPIs
   - Overall attrition status
   - Management-level overview

2. **Operational Analysis**
   - Attrition by department and job role
   - Overtime and tenure analysis
   - Compensation and employee-segment comparisons

3. **Predictive Insights**
   - Model performance
   - Key predictors
   - High-risk employee segments

The dashboard follows an inverted-pyramid layout, presents headline information first, and supports cross-filtering and drill-through.

### Dashboard Preview

Add a dashboard screenshot to the `docs` folder and use:

```markdown
![Employee Attrition Dashboard](docs/dashboard-preview.png)
```

## Business Recommendations

1. **Reduce chronic overtime**  
   Audit workloads and staffing levels in teams with consistently high overtime.

2. **Strengthen early-tenure support**  
   Introduce structured onboarding, check-ins, and career guidance during the first two years.

3. **Review high-risk roles**  
   Investigate workload, compensation, management support, and career progression for Sales Representatives and other high-attrition roles.

4. **Conduct targeted compensation reviews**  
   Focus on high-risk roles and employees in the lowest income quartile instead of applying blanket salary increases.

5. **Operationalize the model carefully**  
   Use model probabilities as part of a monthly HR watch list, with human review and appropriate privacy and fairness controls.

6. **Monitor retention KPIs**  
   Update the KPI dashboard monthly and repeat the analysis quarterly to measure the effect of interventions.

7. **Improve source-system data quality**  
   Use validation lists, uniqueness checks, correct data types, and automated range controls.

## Tools and Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib / Seaborn
- Microsoft Excel
- Power BI

## Repository Structure

```text
.
├── data/                 # Raw and cleaned datasets, if redistribution is allowed
├── notebooks/            # Cleaning, EDA, and modeling notebooks
├── dashboard/            # Power BI files
├── docs/                 # Dashboard previews and documentation
├── reports/              # Final project report
├── src/                  # Reusable Python scripts
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

Update the structure above to match the actual repository files.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/employee-attrition-analytics.git
cd employee-attrition-analytics
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the analysis

```bash
jupyter notebook
```

Open the notebooks in workflow order and run the cells sequentially.

### 5. Open the dashboard

Open the `.pbix` file in Power BI Desktop and update the local dataset path if required.

## Limitations and Responsible Use

- The dataset is synthetic and may not fully represent a real organization.
- Data-quality issues were deliberately injected into a working copy for educational purposes.
- Model recall is limited, so the model does not identify every employee who leaves.
- Predictive relationships do not establish causality.
- Employment-related predictions can create fairness, privacy, and discrimination risks.
- Sensitive or demographic variables should not be used to make automatic employment decisions.
- Any practical deployment should include human oversight, access controls, bias testing, and ongoing monitoring.

## Future Improvements

- Tune classification thresholds to improve recall while controlling false positives.
- Address class imbalance using class weights or appropriate resampling methods.
- Add SHAP-based model explanations.
- Perform fairness analysis across relevant employee groups.
- Add hire dates and location data for time-series and geographic analysis.
- Evaluate the effect of overtime-reduction interventions over time.
- Deploy a secure dashboard with role-based access controls.
- Monitor model drift and retrain the model periodically.

## Acknowledgment

This project was developed as a final data analytics, data science, and business intelligence project using the IBM HR Analytics Employee Attrition & Performance dataset.

## License

This repository is intended for educational and portfolio purposes. Add an open-source license only after confirming that all datasets, reports, dashboard assets, and code can be legally redistributed.
