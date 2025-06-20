# app.py

import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
import numpy as np # Needed if processing data as numpy arrays

# Initialize the Flask app
app = Flask(__name__)

# Define the filenames for the saved model and scaler
model_filename = 'shelf_life_prediction_model_rf.pkl'
scaler_filename = 'scaler.pkl'

# Load the saved model and scaler
# Use a try-except block for robust loading
try:
    loaded_model = joblib.load(model_filename)
    loaded_scaler = joblib.load(scaler_filename)
    print("Model and Scaler loaded successfully.")
except FileNotFoundError:
    print(f"Error: Ensure '{model_filename}' and '{scaler_filename}' are in the same directory as app.py")
    # Depending on your deployment strategy, you might want to exit or raise an error here
    loaded_model = None
    loaded_scaler = None
except Exception as e:
    print(f"An error occurred while loading model or scaler: {e}")
    loaded_model = None
    loaded_scaler = None


# Define the home page route that serves the index.html file
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route (e.g., for receiving form data via POST)
@app.route('/predict', methods=['POST'])
def predict():
    if loaded_model is None or loaded_scaler is None:
        return jsonify({'error': 'Model or scaler not loaded'}), 500 # Return an error if loading failed

    # Get data from the POST request
    # Assuming the input comes as JSON from the form or API call
    data = request.form # Use request.form for form data

    # Extract features from the form data
    try:
        # Convert form data to a pandas DataFrame
        # Ensure the column names and order match your training data
        input_data = pd.DataFrame({
            'Moisture (%)': [float(data['moisture'])],
            'Storage Temp (°C)': [float(data['temp'])],
            'RH (%)': [float(data['rh'])],
            'Initial FFA (%)': [float(data['ffa'])],
            'Initial Microbial Count (log CFU/g)': [float(data['microbial'])]
        })
    except (KeyError, ValueError) as e:
        return jsonify({'error': f'Invalid input data. Missing key or incorrect format: {e}'}), 400 # Bad Request


    # Scale the input data
    # The scaler expects input in the same format as the training data (2D array-like)
    scaled_input_data = loaded_scaler.transform(input_data)

    # Make prediction
    prediction = loaded_model.predict(scaled_input_data)

    # Return the prediction as JSON
    # predicted_shelf_life is a numpy array, get the first element
    predicted_shelf_life = prediction[0]

    # Return as JSON
    return jsonify({'predicted_shelf_life_days': float(predicted_shelf_life)}) # Convert numpy float to Python float


# Run the Flask app (for local testing)
# On production servers, a WSGI server like Gunicorn or uWSGI is used
if __name__ == '__main__':
    # app.run(debug=True) # Use debug=True for local development
    app.run(host='0.0.0.0', port=5000) # Run on all available interfaces and port 5000
