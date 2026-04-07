import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, f1_score, recall_score,
                             classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

# 1. Load the dataset
df = pd.read_csv('ai4i2020.csv')

# 2. Feature Engineering
# Physics-based feature: Thermal stress indicator
df['Temperature Difference'] = df['Process temperature [K]'] - df['Air temperature [K]']
# Physical feature: Mechanical load indicator
df['Power'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']

# 3. Data Preparation
X = df.drop(columns=['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
y = df['Machine failure']

# Encode categorical 'Type'
le = LabelEncoder()
X['Type'] = le.fit_transform(X['Type'])

# 4. Stratified Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Handle Class Imbalance (Oversampling)
# We balance the training set to help the model learn the failure patterns
train_data = pd.concat([X_train, y_train], axis=1)
df_majority = train_data[train_data['Machine failure'] == 0]
df_minority = train_data[train_data['Machine failure'] == 1]

df_minority_oversampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_resampled = pd.concat([df_majority, df_minority_oversampled])

X_train_res = df_resampled.drop('Machine failure', axis=1)
y_train_res = df_resampled['Machine failure']

# 6. Pipeline Construction
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 7. Hyperparameter Tuning with GridSearchCV
# Focus: Maximizing Recall for the failure class (class 1)
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='recall', # Optimize for recall of the positive class
    cv=3,
    n_jobs=-1
)

grid_search.fit(X_train_res, y_train_res)

# 8. Best Model and Evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"Best Parameters: {grid_search.best_params_}")
print("\nNew Classification Report:\n", classification_report(y_test, y_pred))

# 9. Visualizations
# A. Confusion Matrix Bar Chart
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
cm_labels = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
cm_values = [tn, fp, fn, tp]

plt.figure(figsize=(10, 6))
bars = plt.bar(cm_labels, cm_values, color=['#2ecc71', '#e74c3c', '#f1c40f', '#3498db'])
plt.title('Optimized Confusion Matrix: Classification Counts', fontsize=14)
plt.ylabel('Count')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', fontweight='bold')
plt.savefig('optimized_confusion_matrix.png')

# B. Feature Importance Chart
importances = best_model.named_steps['classifier'].feature_importances_
feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(feat_df['Feature'], feat_df['Importance'], color='#8e44ad')
plt.title('Optimized Sensor Importance', fontsize=14)
plt.xlabel('Importance Score')
for index, value in enumerate(feat_df['Importance']):
    plt.text(value + 0.005, index, f'{value:.4f}', va='center')
plt.tight_layout()
plt.savefig('optimized_feature_importance.png')