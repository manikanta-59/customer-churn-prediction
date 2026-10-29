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

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Logistic Regression
- Streamlit
- Joblib

## Project Workflow

1. Load and inspect the customer dataset.
2. Convert `TotalCharges` to numeric values and inspect missing data.
3. Explore churn distribution and customer characteristics.
4. Separate input features from the target variable.
5. Split data into training and testing sets.
6. Preprocess numerical and categorical features using a scikit-learn pipeline.
7. Train a Logistic Regression model.
8. Evaluate predictions on held-out test data.
9. Save the trained model and serve predictions through Streamlit.

## Model Evaluation

The initial Logistic Regression model produced these results on a held-out test set:

| Metric | Result |
|---|---:|
| Accuracy | 80.55% |
| Precision — churn class | 65.72% |
| Recall — churn class | 55.88% |
| F1-score — churn class | 60.40% |

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

The estimated probability is a model output, not a guarantee of future customer behaviour.

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
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
