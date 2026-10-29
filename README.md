# Customer Churn Prediction Using Machine Learning

## Project Overview

This project predicts whether a telecommunications customer is likely to leave a company using machine learning. It includes data cleaning, exploratory data analysis, model training, evaluation, and an interactive web application built with Streamlit.

## Problem Statement

Customer churn can affect business revenue and customer retention. The objective is to use customer information to identify potential churners and help businesses investigate customer retention opportunities.

## Dataset

- **Dataset:** IBM Telco Customer Churn
- **Source:** [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records:** 7,043 customers
- **Original columns:** 21
- **Target variable:** `Churn`

The dataset contains customer demographics, service subscriptions, contract details, tenure, monthly charges, total charges, and churn status.

## ## Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **ML Algorithms:** Logistic Regression, Random Forest
- **Data Preprocessing:** SimpleImputer, StandardScaler, OneHotEncoder, ColumnTransformer, Pipeline
- **Model Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrix
- **Model Persistence:** Joblib
- **Web Application:** Streamlit
- **Automated Testing:** Pytest

## ## Project Workflow

1. Load and inspect the IBM Telco Customer Churn dataset.
2. Clean the data and convert `TotalCharges` to numeric values.
3. Perform exploratory data analysis (EDA) to understand churn patterns.
4. Separate input features from the target variable.
5. Split the dataset into training and testing sets using stratified sampling.
6. Build preprocessing pipelines for numerical and categorical features.
7. Train a Logistic Regression classification model.
8. Train a Random Forest classification model for comparison.
9. Evaluate both models using accuracy, precision, recall, and F1-score.
10. Select Logistic Regression as the current candidate based on the comparison results.
11. Save the trained Logistic Regression pipeline using Joblib.
12. Build an interactive Streamlit application to predict customer churn.
13. Write automated tests using Pytest to check model loading, predictions, and prediction probabilities.

## Model Evaluation

The initial Logistic Regression model produced these results on a held-out test set:

| Metric | Result |
|---|---:|
| Accuracy | 80.55% |
| Precision — churn class | 65.72% |
| Recall — churn class | 55.88% |
| F1-score — churn class | 60.40% |

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.55% | 65.72% | 55.88% | 60.40% |
| Random Forest | 77.57% | 60.00% | 46.52% | 52.41% |

Logistic Regression performed better across all four metrics
on this train/test split and was selected as the current
candidate model.

### Confusion Matrix

| Actual / Predicted | No Churn | Churn |
|---|---:|---:|
| No Churn | 926 | 109 |
| Churn | 165 | 209 |

These results represent an initial baseline. The model detects some customers who churn, but it also misses others. Further evaluation and improvement would be required before using it for real business decisions.

## Application Features

- Interactive customer input form.
- Estimated churn probability.
- Predicted churn or no-churn classification.
- Local web interface powered by Streamlit.

## Decision Threshold Experiment

The Logistic Regression model was evaluated at different classification thresholds using the same held-out test set.

| Threshold | Accuracy | Precision | Recall | F1 Score |
|---:|---:|---:|---:|---:|
| 0.50 | 80.55% | 65.72% | 55.88% | 60.40% |
| 0.40 | 77.79% | 56.95% | 66.84% | 61.50% |
| 0.35 | 76.51% | 54.43% | 70.59% | 61.47% |
| 0.30 | 74.95% | 51.93% | 75.40% | 61.50% |

### Observations

- Lowering the threshold increases recall, allowing the model to identify more customers who are likely to churn.
- Precision decreases at lower thresholds because more customers who would not churn are incorrectly flagged.
- The F1 score remains nearly the same across thresholds 0.40, 0.35, and 0.30.

### Selected Threshold

A threshold of **0.40** was selected as a provisional operating point because it improves recall while maintaining a reasonable balance between precision and recall.

The final threshold should depend on business requirements and the relative costs of false positives and false negatives. These results are based on the current held-out test set and may not generalize to new data.

The estimated probability is a model output, not a guarantee of future customer behaviour.

## Installation and Setup

### ### 1. Clone the repository

```bash
git clone https://github.com/manikanta-59/customer-churn-prediction.git
cd customer-churn-prediction
```

Replace the placeholder with your actual repository URL after uploading the project.

### 2. Create and activate a virtual environment

On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Obtain the dataset

Download the dataset from the Kaggle source above. Place the CSV file in the project root with the filename:

`WA_Fn-UseC_-Telco-Customer-Churn.csv`

Check the dataset licence and usage terms before redistributing it.

### 5. Train and save the model

```bash
python train_model.py
```

This trains the model, prints evaluation metrics, and creates `churn_model.joblib`.

### 6. Launch the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Limitations and Future Improvements

- Compare Logistic Regression with other suitable classification algorithms.
- Evaluate class imbalance and decision thresholds.
- Improve churn recall while monitoring precision.
- Add automated tests and stronger input validation.
- Add model monitoring and reproducible experiment tracking.
- Deploy the application to a suitable hosting service.

## Disclaimer

This is an educational machine learning portfolio project. Predictions should not be treated as guarantees or used for consequential business decisions without additional validation.
