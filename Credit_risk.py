import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
data = pd.read_csv("D:/SEM-3/ML/ML Assignment/Credit_Risk_Classification - Credit_Risk_Classification.csv")  

# Encode categorical columns
le = LabelEncoder()
data['default_history'] = le.fit_transform(data['default_history'])  # Yes=1, No=0
data['credit_risk'] = le.fit_transform(data['credit_risk'])          # Low=1, Medium=2, High=0

# Feature & target split
X = data.drop(['applicant_id', 'credit_risk'], axis=1)
y = data['credit_risk']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


# ------------------ Decision Tree ------------------
print("\n Decision Tree")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
dt_proba = dt.predict_proba(X_test)

dt_acc = accuracy_score(y_test, dt_pred)
dt_prec = precision_score(y_test, dt_pred, average='macro')
dt_rec = recall_score(y_test, dt_pred, average='macro')
dt_f1 = f1_score(y_test, dt_pred, average='macro')
dt_logloss = log_loss(y_test, dt_proba)

print(f"Accuracy: {dt_acc:.2f}")
print(f"Precision: {dt_prec:.2f}")
print(f"Recall: {dt_rec:.2f}")
print(f"F1 Score: {dt_f1:.2f}")
print(f"Log Loss: {dt_logloss:.2f}")

dt_cm = confusion_matrix(y_test, dt_pred)
ConfusionMatrixDisplay(dt_cm).plot()
plt.title("Confusion Matrix - Decision Tree")
plt.show()

# ------------------ Random Forest ------------------
print("\n Random Forest")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred, average='macro')
rf_rec = recall_score(y_test, rf_pred, average='macro')
rf_f1 = f1_score(y_test, rf_pred, average='macro')
rf_logloss = log_loss(y_test, rf_proba)

print(f"Accuracy: {rf_acc:.2f}")
print(f"Precision: {rf_prec:.2f}")
print(f"Recall: {rf_rec:.2f}")
print(f"F1 Score: {rf_f1:.2f}")
print(f"Log Loss: {rf_logloss:.2f}")

rf_cm = confusion_matrix(y_test, rf_pred)
ConfusionMatrixDisplay(rf_cm).plot()
plt.title("Confusion Matrix - Random Forest")
plt.show()


# ------------------ Logistic Regression ------------------
print("\n Logistic Regression")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
lr_proba = lr.predict_proba(X_test)

lr_acc = accuracy_score(y_test, lr_pred)
lr_prec = precision_score(y_test, lr_pred, average='macro')
lr_rec = recall_score(y_test, lr_pred, average='macro')
lr_f1 = f1_score(y_test, lr_pred, average='macro')
lr_logloss = log_loss(y_test, lr_proba)

print(f"Accuracy: {lr_acc:.2f}")
print(f"Precision: {lr_prec:.2f}")
print(f"Recall: {lr_rec:.2f}")
print(f"F1 Score: {lr_f1:.2f}")
print(f"Log Loss: {lr_logloss:.2f}")

lr_cm = confusion_matrix(y_test, lr_pred)
ConfusionMatrixDisplay(lr_cm).plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# ------------------ Visualizations & Final Comparison ------------------

# Compare metrics in a DataFrame
results = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest', 'Logistic Regression'],
    'Accuracy': [dt_acc, rf_acc, lr_acc],
    'Precision': [dt_prec, rf_prec, lr_prec],
    'Recall': [dt_rec, rf_rec, lr_rec],
    'F1 Score': [dt_f1, rf_f1, lr_f1],
    'Log Loss': [dt_logloss, rf_logloss, lr_logloss]
})

print("\n Model Comparison:")
print(results.set_index('Model'))

# Barplot for metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
plt.figure(figsize=(10,6))
results_melted = results.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Score')
sns.barplot(data=results_melted, x='Metric', y='Score', hue='Model')
plt.title("Model Performance Comparison")
plt.ylim(0, 1)
plt.legend(title='Model')
plt.show()

# Select best model based on F1 Score (or any metric you prefer)
best_idx = results['F1 Score'].idxmax()
best_model = results.loc[best_idx, 'Model']
print(f"\n Best Model: {best_model} (F1 Score = {results.loc[best_idx, 'F1 Score']:.2f})")