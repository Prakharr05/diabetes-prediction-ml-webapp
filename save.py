import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the diabetes dataset
data = pd.read_csv('diabetes.csv')

# Assuming the last column is the target variable
X = data.iloc[:, :-1]  # Features (all columns except the last one)
y = data.iloc[:, -1]   # Labels (the last column)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest model 
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "diabetes_detection_model.pkl")
print("Model saved as 'diabetes_detection_model.pkl'")

# Load the model (for future use)
# model = joblib.load("diabetes_detection_model.pkl")