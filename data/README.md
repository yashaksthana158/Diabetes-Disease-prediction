# Data Directory

## Dataset: Pima Indians Diabetes Dataset

| Property | Value |
|---|---|
| **File** | `diabetes.csv` |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) / UCI ML Repository |
| **Rows** | 768 |
| **Features** | 8 clinical measurements |
| **Target** | `Outcome` — 0 (Non-Diabetic) · 1 (Diabetic) |

## Download Instructions

This file is excluded from version control (see `.gitignore`).

1. Go to: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
2. Download `diabetes.csv`
3. Place it in this directory: `data/diabetes.csv`

## Feature Reference

| Column | Description | Missing Values Strategy |
|---|---|---|
| `Pregnancies` | Number of times pregnant | None (0 is valid) |
| `Glucose` | Plasma glucose (mg/dL) | Zero → median imputation |
| `BloodPressure` | Diastolic BP (mm Hg) | Zero → median imputation |
| `SkinThickness` | Triceps skin fold (mm) | Zero → median imputation |
| `Insulin` | 2-hr serum insulin (μU/mL) | Zero → median imputation |
| `BMI` | Body mass index (kg/m²) | Zero → median imputation |
| `DiabetesPedigreeFunction` | Family history score | None |
| `Age` | Patient age (years) | None |
| `Outcome` | **Target** — 0 or 1 | — |
