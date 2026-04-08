# 🔧 Predictive Maintenance using AI-Generated Machine Learning

## Overview

This project investigates the use of AI-generated code in developing a machine learning (ML) model for predicting industrial machine failures. The study focuses on evaluating the quality of the initial AI-generated solution, identifying its limitations, and improving it through iterative development.

The project demonstrates how AI-generated code can act as a useful starting point, but requires human refinement to produce a more reliable and well-structured model.


## Dataset

The dataset used in this project is the **AI4I 2020 Predictive Maintenance Dataset**, a synthetic dataset simulating real industrial sensor readings.

It contains features such as:

- Air temperature (K)
- Process temperature (K)
- Rotational speed (rpm)
- Torque (Nm)
- Tool wear (min)

The target variable is binary:
- `0` → No machine failure
- `1` → Machine failure

## Models Used

- Random Forest Classifier (baseline model)
- Random Forest with Oversampling & Feature Engineering
- Tuned Random Forest (using GridSearchCV)
- Safety-First Random Forest (adjusted classification threshold)

## Development Process

The project was developed using version control to track the progression from the initial AI-generated code to the final improved model.

Key stages included:

1. Initial AI-generated Random Forest classifier
2. Feature engineering (Temperature Difference and Power columns)
3. Class imbalance handling using Random Oversampling
4. Improved evaluation using Balanced Accuracy and F1-Score
5. Hyperparameter tuning using GridSearchCV
6. Classification threshold adjustment using Precision-Recall Curve

## Results

| Model | Balanced Accuracy | Recall (Failure) | Precision (Failure) | F1 (Failure) |
|---|---|---|---|---|
| Initial Model (V1) | — | 0.59 | 0.84 | 0.69 |
| Pipeline & Physics (V2) | 86.64% | 0.74 | 0.91 | 0.81 |
| Optimised Model (V3) | 86.64% | 0.74 | 0.91 | 0.81 |
| Safety-First Model (V4) | 91.79% | 0.85 | 0.64 | 0.73 |

Best parameters identified by GridSearchCV: confirmed default settings were already optimal for this dataset.

## Key Observations

- The initial model achieved 98.40% accuracy but had a critical flaw — a Recall of only 0.59 for failures due to class imbalance
- Feature engineering and oversampling significantly improved failure detection
- Hyperparameter tuning confirmed default settings were well-suited to this dataset
- Lowering the classification threshold to 0.24 improved Recall to 0.85, reducing missed failures from 26 to 10

## Evaluation Metrics

The models were evaluated using:

- Balanced Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-Score

Particular attention was given to Recall, as false negatives (missed machine failures) are critical in industrial applications where undetected faults can cause equipment damage or risk worker safety.

## Limitations

- The dataset is synthetic and may not fully represent real-world industrial conditions
- The model does not incorporate time-series or sequential sensor data
- 15% of failures are still missed in the final model
- Results may not generalise to machines outside the simulated environment

## Key Insights

- AI-generated code provides a useful starting point but is often incomplete
- Human intervention is necessary to ensure proper evaluation and optimisation
- Accuracy alone is a misleading metric in imbalanced classification problems
- Evaluation metrics beyond accuracy are essential in safety-critical domains

## Technologies Used

- Python
- pandas
- scikit-learn
- matplotlib

## Repository Structure

```
├── data/
│ └── ai4i2020.csv
├── PMM_Initial.py                  # V1 – Baseline model
├── PMM_v2_Pipeline_Physics.py      # V2 – Pipeline & feature engineering
├── PMM_v3_Optimisation.py          # V3 – Hyperparameter tuning
├── PMM_v4_SafetyFirst.py           # V4 – Threshold adjustment
├── README.md
└── requirements.txt
```

## How to Run the Project

1. Install dependencies: `pip install -r requirements.txt`
2. Download dataset: `ai4i2020.csv`
3. Run any version: `python PMM_v4_SafetyFirst.py`

Each script will print evaluation metrics to the console and save figures as `.png` files in the working directory.

## Conclusion

This project highlights the importance of critically evaluating AI-generated code. While AI tools can quickly produce functional solutions, they require careful review, improvement, and validation to ensure reliability and effectiveness — particularly in safety-critical industrial environments.

The final model demonstrates improved performance through iterative refinement, reinforcing the role of human oversight in machine learning workflows.
