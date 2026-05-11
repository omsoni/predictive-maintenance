# Wind Turbine Predictive Maintenance — Deep Learning Classifier

## Overview

Binary classification of wind turbine generator failure from 40 anonymized sensor features (20,000 training / 5,000 test observations), with severe class imbalance (~95% no-failure, ~5% failure).

The notebook iterates through **six feedforward neural network architectures** — from a single-hidden-layer SGD baseline to deeper networks with BatchNorm, Dropout, He initialization, AdamW, L2 regularization, focal loss, and learning-rate scheduling — and selects the final model using a **cost-sensitive evaluation framework** that weights false negatives at 10× the cost of false positives (reflecting the business reality that missed failures lead to replacement costs while false alarms lead to inspection costs).

## Final Model

**Model 2** — two hidden layers, BatchNorm + Dropout, Adam optimizer, class-weighted Binary Crossentropy loss.

| Metric | Value |
|---|---|
| Overall Accuracy | ~99% |
| Failure Recall (Class 1) | ~88% |
| Total Business Cost (Test Set) | Lowest among candidates |

## Dataset

- **Features:** 40 anonymized continuous sensor variables (V1–V40)
- **Target:** Binary — `1` = failure, `0` = no failure
- **Training set:** 20,000 observations
- **Test set:** 5,000 observations
- **Class imbalance:** ~95% / 5%

## Techniques Covered

- Stratified handling of imbalanced data
- Class weighting for loss balancing
- Median imputation (fit on training set only)
- Standard scaling for ReLU-based networks
- Univariate AUC analysis for feature ranking
- Correlation analysis for feature redundancy
- Threshold tuning via cost minimization
- Comparative training-curve analysis across architectures

## Tech Stack

- TensorFlow / Keras
- scikit-learn
- pandas, NumPy
- matplotlib, seaborn

## Repository Structure

```
.
├── Wind_Turbine_Predictive_Maintenance.ipynb   # Main notebook
├── Train.csv                                   # Training data
├── Test.csv                                    # Test data
└── README.md
```
