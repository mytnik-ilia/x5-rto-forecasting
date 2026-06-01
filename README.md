# X5 Retail RTO Forecasting

Advanced Machine Learning pipeline for retail turnover (RTO) forecasting using time-series feature engineering, gradient boosting ensembles and hyperparameter optimization.

---

# Project Overview

This project focuses on predicting retail turnover (RTO) using tabular retail data and historical time-series patterns.

The solution was designed in a competitive Kaggle-style environment with emphasis on:

* advanced feature engineering
* time-series forecasting
* leakage-safe validation
* boosting model optimization
* ensemble learning
* production-style inference pipeline

The repository demonstrates practical Machine Learning engineering techniques commonly used in high-performance tabular competitions.

---

# Problem Statement

The task is to forecast retail turnover for stores using:

* historical sales behavior
* promotional activity
* customer metrics
* competition density
* temporal dynamics
* operational store features

The primary objective was maximizing prediction accuracy through robust feature engineering and ensemble boosting methods.

---

# Pipeline Architecture

The final pipeline includes:

```text id="x4r9st"
Raw Data
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Time-Series Features
   ↓
Target Transformations
   ↓
Cross Validation
   ↓
Boosting Models
   ↓
Model Ensemble
   ↓
Inference Pipeline
```

---

# Models Used

The final solution combines several gradient boosting frameworks:

## CatBoost

Main high-performance boosting model with categorical feature handling.

## LightGBM

Optimized gradient boosting with Optuna tuning.

## XGBoost

Additional ensemble diversity and robust error minimization.

---

# Hyperparameter Optimization

Hyperparameter tuning was performed using:

* Optuna
* cross-validation optimization
* fold-aware evaluation
* automated parameter search

Optimized parameters include:

* learning rate
* max depth
* regularization
* feature subsampling
* leaf constraints
* boosting iterations

---

# Feature Engineering

A major focus of the project was advanced feature engineering.

## Time-Series Features

Implemented features:

* lag features
* rolling window statistics
* moving averages
* trend indicators
* historical aggregations

Examples:

```python id="0o3u1d"
lag_1
lag_2
lag_3
lag_6
lag_12
```

Rolling statistics:

```python id="n4j1pi"
roll_mean
roll_std
roll_min
roll_max
roll_median
```

---

# Interaction Features

Custom interaction features were engineered to improve signal extraction.

Examples:

```python id="c1f1s9"
avg_promo_items × foot_traffic
avg_items_in_check × working_hours
promo × traffic
competition × store_activity
```

---

# Competition Features

Competition-aware metrics were introduced to capture local retail density.

Examples:

```python id="q9s1lz"
grocery_500m + pyaterochka_500m
```

These features help model competitive pressure between nearby stores.

---

# Target Engineering

The solution includes advanced target transformations:

* target ratio normalization
* log-transformed targets
* clipped target distributions
* historical target statistics

This significantly improved model stability.

---

# Validation Strategy

Special attention was paid to preventing target leakage.

Implemented techniques:

* time-series aware validation
* month-based holdout strategy
* fold-safe preprocessing
* train/validation temporal separation
* inference-safe feature generation

Validation was performed using the latest available month as holdout data.

---

# Ensemble Strategy

The final prediction pipeline combines:

* CatBoost predictions
* LightGBM predictions
* XGBoost predictions
* log-target boosting models
* statistical baseline models

Additional ensemble components:

* rolling mean baselines
* seasonal historical averages
* last-value estimators

This ensemble significantly improved prediction robustness.

---

# Repository Structure

```text id="m6h8tz"
x5-rto-forecasting/
│
├── notebooks/
│   └── x5_rto_solution.ipynb
│
├── src/
│   ├── features.py
│   ├── train.py
│   ├── inference.py
│   └── utils.py
│
├── data/
│   └── README.md
│
├── submissions/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Technologies

## Core Stack

* Python
* Pandas
* NumPy
* Scikit-learn

## Boosting Frameworks

* CatBoost
* LightGBM
* XGBoost

## Optimization

* Optuna

## Visualization

* Matplotlib

---

# Key ML Concepts Demonstrated

This project demonstrates practical experience with:

* feature engineering
* time-series ML
* gradient boosting
* ensemble methods
* hyperparameter optimization
* leakage prevention
* production-style inference
* validation strategy design
* retail forecasting
* tabular machine learning

---

# Results

The final solution significantly improved baseline performance through:

* advanced lag engineering
* rolling statistical features
* ensemble boosting
* optimized validation strategy
* target transformations
* competition-aware features

| Model                   | Score |
| ----------------------- | ----- |
| Baseline                | 0.88  |
| Improved Features       | 0.94  |
| Final Ensemble Pipeline | 0.96+ |

---

# Installation

Clone the repository:

```bash id="gk0g6r"
git clone https://github.com/YOUR_USERNAME/x5-rto-forecasting.git
cd x5-rto-forecasting
```

Install dependencies:

```bash id="n0y5y6"
pip install -r requirements.txt
```

---

# Run Training

```bash id="b4h9h4"
python train.py
```

---

# Run Inference

```bash id="wh7g7x"
python inference.py
```

---

# Future Improvements

Potential future improvements:

* stacking/blending ensembles
* SHAP interpretability
* automated feature generation
* GPU optimization
* distributed training
* neural tabular architectures
* online inference pipeline

---

# Author

Machine Learning / Data Science project focused on advanced retail forecasting, feature engineering and high-performance tabular ML pipelines.
