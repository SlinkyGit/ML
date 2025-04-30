# Closed-Form Linear Regression with Cross-Validation

This project implements a linear regression model using the **closed-form solution** (normal equation) on a health insurance dataset. It includes preprocessing steps, model training and evaluation, and a detailed cross-validation routine.

## Overview

The goal of this project is to:
- Prepare and encode a real-world dataset (`insurance.csv`)
- Apply closed-form linear regression
- Evaluate model performance using RMSE and SMAPE metrics
- Implement **k-fold cross-validation** with variable fold sizes

## Dataset

The dataset used (`insurance.csv`) includes the following features:
- `age`, `sex`, `bmi`, `children`, `smoker`, `region`
- Target variable: `charges` (medical cost)

Ensure the CSV file is located in the same directory as the Python script for the code to run properly.

## Key Features

### 🔧 Preprocessing
- Binary encoding for `sex` and `smoker`
- One-hot encoding for `region`
- Bias term added manually
- Data split into training (2/3) and validation (1/3) sets

### 📈 Closed-Form Linear Regression
- Uses the normal equation to calculate weights:
```bash
Training RMSE: <value> Validation RMSE: <value> Training SMAPE: <value> Validation SMAPE: <value>
Cross-Validation results for S = 3: [mean_RMSE, std_dev]
Cross-Validation results for S = 223: [mean_RMSE, std_dev]
Cross-Validation results for S = 1338: [mean_RMSE, std_dev]
```


## How to Run
1. Clone this repository
2. Place `insurance.csv` in the project directory
3. Run the script using Python 3

```bash
python3 lin-reg.py
