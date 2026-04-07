import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, f1_score, 
                             classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE 

# 1. Load the dataset
df = pd.read_csv('ai4i2020.csv')

# 2. Feature Engineering
# Physics-based relationship: Temperature difference between process and environment
df['Temperature Difference'] = df['Process temperature [K]'] - df['Air temperature [K]']
# Physical relationship: Mechanical Power is proportional to Torque * Speed
df['Power'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']

# 3. Data Preparation
# Dropping ID columns and specific failure modes to prevent data leakage
X = df.drop(columns=['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
y = df['Machine failure']

# Encode categorical 'Type' (L, M, H)
le = LabelEncoder()
X['Type'] = le.fit_transform(X['Type'])

# 4. Train/Test Split (Stratified to maintain 3% failure ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Handle Class Imbalance
# Using oversampling to balance the failure class (Minority: ~3%)
train_data = pd.concat([X_train, y_train], axis=1)
df_majority = train_data[train_data['Machine failure'] == 0]
df_minority = train_data[train_data['Machine failure'] == 1]

df_minority_oversampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_resampled = pd.concat([df_majority, df_minority_oversampled])

X_train_res = df_resampled.drop('Machine failure', axis=1)
y_train_res = df_resampled['Machine failure']

# 6. Pipeline Construction (Scaling + Modeling)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 7. Model Training
pipeline.fit(X_train_res, y_train_res)

# 8. Evaluation
y_pred = pipeline.predict(X_test)
bal_acc = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9. Visualizations
# A. Confusion Matrix Bar Chart
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
cm_labels = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
cm_values = [tn, fp, fn, tp]

plt.figure(figsize=(10, 6))
bars = plt.bar(cm_labels, cm_values, color=['#2ecc71', '#e74c3c', '#f1c40f', '#3498db'])
plt.title('Confusion Matrix: Prediction Distribution', fontsize=14)
plt.ylabel('Count')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', fontweight='bold')
plt.savefig('confusion_matrix_bar.png')

# B. Feature Importance
importances = pipeline.named_steps['classifier'].feature_importances_
feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(feat_df['Feature'], feat_df['Importance'], color='#34495e')
plt.title('Predictive Sensor Importance', fontsize=14)
plt.xlabel('Importance Score')
for index, value in enumerate(feat_df['Importance']):
    plt.text(value + 0.005, index, f'{value:.4f}', va='center')
plt.tight_layout()
plt.savefig('feature_importance.png')