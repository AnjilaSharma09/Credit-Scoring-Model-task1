# Credit-Scoring-Model-task1
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
# For demonstration purposes, let's assume we have a CSV dataset called 'credit_data.csv'
# It should contain features such as 'credit_history', 'income', 'debt', and 'target' (default or not)
data = pd.read_csv('credit_data.csv')

# Display the first few rows of the dataset
print(data.head())

# Feature selection (assuming we have columns: 'credit_history', 'income', 'debt', etc.)
# Separating the independent variables (X) and the target variable (y)
X = data[['credit_history', 'income', 'debt', 'loan_amount', 'age']]
y = data['target']  # 1 for default, 0 for non-default

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data (important for models like Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print classification report for detailed performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

