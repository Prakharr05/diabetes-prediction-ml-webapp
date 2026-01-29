from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the model
model = joblib.load("diabetes_detection_model.pkl")

# Use a default StandardScaler if 'scaler.pkl' is missing
try:
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    print("Warning: scaler.pkl not found. Using a new StandardScaler instance.")
    scaler = StandardScaler()  # Creates a new scaler, but it won’t be trained

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data
        data = request.get_json()
        input_data = np.array([
            data["Pregnancies"], data["Glucose"], data["BloodPressure"],
            data["SkinThickness"], data["Insulin"], data["BMI"],
            data["DiabetesPedigreeFunction"], data["Age"]
        ]).reshape(1, -1)
        
        # Standardize the input (if scaler is not trained, it will do nothing)
        try:
            input_data = scaler.transform(input_data)
        except AttributeError:
            pass  # If scaler is untrained, skip transformation
        
        # Make a prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        
        # Return the result
        result = {
            "prediction": int(prediction[0]),
            "probability": float(probability[0][1])
        }
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
