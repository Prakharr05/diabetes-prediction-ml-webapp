import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
data = pd.read_csv(url, names=column_names)

# Check for missing values
print("Missing values in the dataset:")
print(data.isnull().sum())

# Split features and target
X = data.drop("Outcome", axis=1)  # Features
y = data["Outcome"]  # Target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Random Forest model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "diabetes_detection_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved.")

# Initialize an online learning model
online_model = SGDClassifier(loss='log_loss', random_state=42, max_iter=1000, tol=1e-3)
online_model.partial_fit(X_train, y_train, classes=np.unique(y))

# Generate predictions on the test set
y_pred = model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Diabetes Prediction")
plt.show()

# Feature Importance
importances = model.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title("Feature Importance for Diabetes Prediction")
plt.show()

# Continuous input loop for new patient data
while True:
    try:
        new_patient = {}
        for feature in feature_names:
            new_patient[feature] = float(input(f"Enter {feature}: "))
        
        new_patient_df = pd.DataFrame([new_patient])
        new_patient_scaled = scaler.transform(new_patient_df)
        
        prediction = model.predict(new_patient_scaled)
        probability = model.predict_proba(new_patient_scaled)
        
        if prediction[0] == 1:
            print("The patient is predicted to be diabetic.")
        else:
            print("The patient is predicted to be non-diabetic.")
        
        print("Probability:", probability)
        
        # Update online model with new data
        online_model.partial_fit(new_patient_scaled, [prediction[0]])
        
    except Exception as e:
        print("Error:", e)
    
    cont = input("Do you want to enter another patient? (yes/no): ")
    if cont.lower() != 'yes':
        break
