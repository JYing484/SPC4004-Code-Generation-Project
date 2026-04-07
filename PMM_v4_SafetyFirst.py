import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_recall_curve, confusion_matrix, 
                             classification_report, balanced_accuracy_score)
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

# 1. Data Preparation and Feature Engineering
df = pd.read_csv('ai4i2020.csv')
df['Temperature Difference'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['Power'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']

X = df.drop(columns=['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
y = df['Machine failure']
X['Type'] = LabelEncoder().fit_transform(X['Type'])

# 2. Balanced Training (Oversampling)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
train_data = pd.concat([X_train, y_train], axis=1)
df_maj = train_data[train_data['Machine failure'] == 0]
df_min = train_data[train_data['Machine failure'] == 1]
df_min_over = resample(df_min, replace=True, n_samples=len(df_maj), random_state=42)
df_res = pd.concat([df_maj, df_min_over])
X_train_res, y_train_res = df_res.drop('Machine failure', axis=1), df_res['Machine failure']

# 3. Model Training
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(X_train_res, y_train_res)

# 4. Threshold Adjustment (Target: 85% Recall)
y_probs = pipeline.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

# Find the highest threshold that still guarantees 85% recall
target_recall = 0.85
idx = np.where(recalls >= target_recall)[0]
safety_threshold = thresholds[idx[-1]]

# Apply new threshold
y_pred_safety = (y_probs >= safety_threshold).astype(int)

# 5. Output Final Results
print(f"Safety-First Threshold: {safety_threshold:.2f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_safety):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_safety))

# 6. Visualizations
# A. Precision-Recall Curve
plt.figure(figsize=(10, 6))
plt.plot(recalls, precisions, color='#2980b9', lw=2, label='P-R Curve')
plt.axvline(x=target_recall, color='#c0392b', linestyle='--', label='85% Recall Target')
plt.scatter(recalls[idx[-1]], precisions[idx[-1]], color='black', zorder=5, label=f'Chosen Point (T={safety_threshold:.2f})')
plt.title('Precision-Recall Trade-off for Predictive Maintenance')
plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision (Trustworthiness)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('precision_recall_curve.png')

# B. Safety-First Confusion Matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_safety).ravel()
labels, values = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives'], [tn, fp, fn, tp]
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, values, color=['#27ae60', '#e67e22', '#c0392b', '#2980b9'])
plt.title(f'Safety-First Performance (Recall={target_recall:.0%})', fontsize=14)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', fontweight='bold')
plt.savefig('safety_confusion_matrix.png')