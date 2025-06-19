import os
import joblib
import pandas as pd
from flask import Flask, request, render_template_string # No need for jsonify if returning string
# import threading # Not used in this snippet
# import time # Not used in this snippet
import numpy as np # Import numpy

# --- Configuration ---
# Assuming you are deploying the scikit-learn model and scaler saved with joblib
MODEL_FILE_NAME = 'best_rf_model.joblib'
SCALER_FILE_NAME = 'scaler.pkl' # Assuming you saved the scaler
INDEX_HTML_FILE_NAME = 'index.html'
FLASK_PORT = int(os.environ.get('PORT', 5000))

# --- Load the Model and Scaler ---
app_dir = os.path.dirname(__file__)
model_path = os.path.join(app_dir, MODEL_FILE_NAME)
scaler_path = os.path.join(app_dir, SCALER_FILE_NAME)

loaded_model = None
loaded_scaler = None

try:
    if os.path.exists(model_path):
        loaded_model = joblib.load(model_path)
        print(f"Model successfully loaded from: {model_path}", flush=True)
    else:
        print(f"Error: Model file not found at {model_path}. Model loading skipped.", flush=True)

    if os.path.exists(scaler_path):
        loaded_scaler = joblib.load(scaler_path)
        print(f"Scaler successfully loaded from: {scaler_path}", flush=True)
    else:
        # This is a critical warning if your model expects scaled data
        print(f"WARNING: Scaler file not found at {scaler_path}. Prediction may fail or be inaccurate if scaling is required.", flush=True)

except FileNotFoundError as e:
    print(f"File not found error during loading: {e}", flush=True)
except Exception as e:
    print(f"An error occurred while loading model or scaler: {e}", flush=True)


# --- Load the HTML Template ---
templates_folder = os.path.join(app_dir, 'templates')
index_html_path = os.path.join(templates_folder, INDEX_HTML_FILE_NAME)

index_html_content = "<html><body><h1>Error loading template.</h1></body></html>"
try:
    if os.path.exists(index_html_path):
        with open(index_html_path, 'r') as f:
            index_html_content = f.read()
        print(f"HTML template loaded from: {index_html_path}", flush=True)
    else:
         print(f"Error: {INDEX_HTML_FILE_NAME} not found at {index_html_path}. Using default error content.", flush=True)

except Exception as e:
    print(f"An error occurred while loading the HTML template: {e}", flush=True)


# --- Flask App Initialization ---
app = Flask(__name__)

# --- Routes ---

@app.route('/')
def index():
    print("Index route accessed.", flush=True)
    return render_template_string(index_html_content)

@app.route('/predict', methods=['POST'])
def predict():
    print("Predict route accessed.", flush=True)

    if loaded_model is None:
        print("Model not loaded, cannot predict.", flush=True)
        return "Error: Model not loaded.", 500

    try:
        # Get data from the POST request form
        storage_temperature = float(request.form['storage_temperature'])
        storage_duration = int(request.form['storage_duration'])
        # Get texture and microbial load as strings first
        texture_str = request.form['texture']
        microbial_load_str = request.form['microbial_load']
        weight_loss = float(request.form['weight_loss'])

        # --- Preprocessing ---
        # Convert texture and microbial_load to float, handle potential errors
        try:
            texture_processed = float(texture_str)
        except ValueError:
            print(f"Warning: Could not convert texture '{texture_str}' to float. Using 0.0.", flush=True)
            texture_processed = 0.0

        try:
            microbial_load_processed = float(microbial_load_str)
        except ValueError:
            print(f"Warning: Could not convert microbial_load '{microbial_load_str}' to float. Using 0.0.", flush=True)
            microbial_load_processed = 0.0
        # *********************************************************

        # Create DataFrame with the correct column names matching training data
        # Ensure the order of columns matches the training data if not using feature names
        # The column names here MUST match the names used when training the scikit-learn model
        input_data = pd.DataFrame([[storage_temperature, storage_duration, texture_processed, microbial_load_processed, weight_loss]],
                                  columns=['Storage Temperature', 'Storage Duration', 'Texture', 'Microbial Load', 'Weight Loss']) # Make sure these match your training feature names

        print(f"Input data DataFrame created: {input_data}", flush=True)

        # --- Scaling ---
        # If a scaler was loaded, scale the input data BEFORE prediction
        if loaded_scaler is not None:
            try:
                # Scikit-learn scalers expect numpy arrays, not DataFrames directly
                # Ensure the column order in the DataFrame matches the order used when fitting the scaler
                scaled_input_array = loaded_scaler.transform(input_data.values) # .values gets the numpy array from DataFrame
                print("Input data scaled.", flush=True)
                # The model expects scaled input
                prediction_input = scaled_input_array
            except Exception as scale_e:
                print(f"Error during scaling: {scale_e}", flush=True)
                return f"An error occurred during scaling: {scale_e}", 500
        else:
            # If scaler is not loaded, use the raw DataFrame values
            # This will likely cause incorrect predictions if your model was trained on scaled data
            print("Scaler not loaded, proceeding with raw input (may cause incorrect predictions).", flush=True)
            prediction_input = input_data.values # Model usually expects numpy array


        # Make prediction
        # Pass the scaled (or raw if no scaler) numpy array to the model
        prediction = loaded_model.predict(prediction_input)[0]

        print(f"Prediction successful: {prediction}", flush=True)
        # Format the output as a string
        return f"Predicted Shelf Life: {prediction:.2f} days"

    except ValueError as ve:
        print(f"ValueError in predict route: {ve}", flush=True)
        return f"Invalid input: {ve}. Please ensure all fields are filled correctly.", 400
    except KeyError as ke:
         print(f"KeyError in predict route (missing form field): {ke}", flush=True)
         return f"Missing form field: {ke}. Please ensure all required fields are in the form.", 400
    except Exception as e:
        print(f"An error occurred during prediction: {e}", flush=True)
        return f"An error occurred during prediction: {e}", 500

# --- Gunicorn Entry Point ---
# Gunicorn typically looks for an 'app' object (your Flask app instance)
if __name__ == '__main__':
    # This block is for running the Flask development server directly
    print(f"App running in development mode on http://127.0.0.1:{FLASK_PORT}", flush=True)
    app.run(port=FLASK_PORT, host='0.0.0.0', debug=True) # Add debug=True for easier development

# Note: When running with Gunicorn, the `if __name__ == '__main__':` block is usually skipped.
# Gunicorn imports your 'app' object directly. Ensure the `app = Flask(__name__)` line
# and the route definitions are at the top level of the script.
