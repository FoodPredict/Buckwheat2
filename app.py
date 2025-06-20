# app.py

import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
import numpy as np
from flask_cors import CORS # <--- Add this import

# Initialize the Flask app
app = Flask(__name__)
CORS(app) # <--- Add this line right after initializing the app

# Define the filenames for the saved model and scaler
# Ensure these paths are correct relative to where app.py is located
model_filename = 'shelf_life_prediction_model_rf.pkl'
scaler_filename = 'scaler.pkl'

# Load the saved model and scaler
# Use a try-except block for robust loading
loaded_model = None
loaded_scaler = None
try:
    loaded_model = joblib.load(model_filename)
    loaded_scaler = joblib.load(scaler_filename)
    print("Model and Scaler loaded successfully.")
except FileNotFoundError:
    print(f"Error: Ensure '{model_filename}' and '{scaler_filename}' are in the same directory as app.py")
    # You might want to handle this more gracefully in production
    # For example, raise an error or exit if essential files are missing
except Exception as e:
    print(f"An error occurred while loading model or scaler: {e}")


# Define the home page route that serves the index.html file
@app.route('/')
def home():
    # Pass necessary data to the template if needed in the future
    return render_template('index.html')

# Define the prediction route (for receiving form data via POST)
@app.route('/predict', methods=['POST'])
def predict():
    # Check if model and scaler were loaded successfully
    if loaded_model is None or loaded_scaler is None:
        # Return an error if model/scaler failed to load at startup
        print("Error: Attempted prediction before model/scaler loaded.")
        return jsonify({'error': 'Prediction service not fully initialized. Model or scaler could not be loaded.'}), 500


    # Get data from the POST request
    # We expect form data from the HTML form
    data = request.form

    # Extract and process features from the form data
    try:
        # Convert form data to a pandas DataFrame
        # Use .get() with default values to handle missing optional inputs
        # Ensure the column names and order match your training data
        input_data = pd.DataFrame({
            'Moisture (%)': [float(data.get('moisture'))],
            'Storage Temp (°C)': [float(data.get('temp'))],
            'RH (%)': [float(data.get('rh'))],
            # Use .get() for optional fields and apply defaults in JS *before* sending
            # If JS default isn't applied, .get() would return None for missing
            # The JS code is already setting defaults, so float() should work
            'Initial FFA (%)': [float(data.get('ffa', '2.0'))], # Use .get() with default as a fallback just in case JS fails
            'Initial Microbial Count (log CFU/g)': [float(data.get('microbial', '5.0'))] # Use .get() with default as a fallback
        })

         # Basic validation for required fields (Moisture, Temp, RH)
        if input_data['Moisture (%)'].isnull().any() or \
           input_data['Storage Temp (°C)'].isnull().any() or \
           input_data['RH (%)'].isnull().any():
             return jsonify({'error': 'Missing required input data (Moisture, Storage Temp, or RH)'}), 400

    except (ValueError, TypeError) as e:
        # Catch errors if float conversion fails (e.g., non-numeric input)
        return jsonify({'error': f'Invalid data format for one or more inputs: {e}'}), 400
    except Exception as e:
        # Catch any other unexpected errors during data processing
        print(f"An unexpected error occurred during data processing: {e}")
        return jsonify({'error': f'An error occurred while processing input data: {e}'}), 500


    # Scale the input data
    try:
        scaled_input_data = loaded_scaler.transform(input_data)
    except Exception as e:
        print(f"An error occurred during data scaling: {e}")
        return jsonify({'error': f'An error occurred during data scaling: {e}'}), 500


    # Make prediction
    try:
        prediction = loaded_model.predict(scaled_input_data)
        # predicted_shelf_life is a numpy array, get the first element and convert to float
        predicted_shelf_life = float(prediction[0])
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': f'An error occurred during prediction: {e}'}), 500


    # Return the prediction as JSON
    # Ensure the response structure matches what the front-end expects
    return jsonify({'predicted_shelf_life_days': predicted_shelf_life})


# Run the Flask app (for local testing)
# On production servers like Render, a WSGI server (Gunicorn) is used
if __name__ == '__main__':
    # Use debug=True for local development to see errors in browser
    # app.run(debug=True, host='0.0.0.0', port=5000)

    # For potential local testing without debug mode, use:
    app.run(host='0.0.0.0', port=5000)
