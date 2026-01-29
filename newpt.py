import pandas as pd
import joblib

# Load the trained model
model = joblib.load("diabetes_detection_model.pkl")

# Example input for a new patient
new_patient = {
    "Pregnancies": 2,
    "Glucose": 150,
    "BloodPressure": 85,
    "SkinThickness": 30,
    "Insulin": 0,
    "BMI": 32,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 45
}

# Convert input to DataFrame
new_patient_df = pd.DataFrame([new_patient])

# Make a prediction (without scaling)
prediction = model.predict(new_patient_df)
probability = model.predict_proba(new_patient_df)

# Output the result
if prediction[0] == 1:
    print("The patient is predicted to be diabetic.")
else:
    print("The patient is predicted to be non-diabetic.")

print("Probability:", probability)