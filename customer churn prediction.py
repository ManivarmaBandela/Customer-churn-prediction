Python_Code
"# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
url = ""https://raw.githubusercontent.com/IBM/telco-customer-churn/master/Telco-Customer-Churn.csv""  # Dataset URL
data = pd.read_csv(url)

# Data Preprocessing
# Display basic information about the dataset
print(data.info())
print(data.describe())
print(data.head())

# Check for missing values
print(""Missing values:
"", data.isnull().sum())

# Data cleaning: Drop unnecessary columns
data.drop(columns=['customerID'], inplace=True)

# Convert target variable to numerical (1 for churn, 0 for not churn)
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

# Convert categorical columns to numerical using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Display the cleaned data
print(data.head())

# Define features and target variable
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
logistic_model = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
logistic_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_logistic = logistic_model.predict(X_test_scaled)

# Evaluation of Logistic Regression
print(""\nLogistic Regression Classification Report:"")
print(classification_report(y_test, y_pred_logistic))
print(""Confusion Matrix:\n"", confusion_matrix(y_test, y_pred_logistic))
print(""Accuracy Score:"", accuracy_score(y_test, y_pred_logistic))

# Random Forest Classifier Model
rf_model = RandomForestClassifier(random_state=42, max_depth=10)  # Added max_depth to reduce overfitting
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluation of Random Forest Classifier
print(""\nRandom Forest Classification Report:"")
print(classification_report(y_test, y_pred_rf))
print(""Confusion Matrix:\n"", confusion_matrix(y_test, y_pred_rf))
print(""Accuracy Score:"", accuracy_score(y_test, y_pred_rf))

# Feature Importance from Random Forest
feature_importances = rf_model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plotting Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature')
plt.title('Feature Importance from Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
"
