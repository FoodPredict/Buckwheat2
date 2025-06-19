from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import tensorflow as tf
import os # Import the os module

app = Flask(__name__)

    # Define model and scaler paths
    MODEL_PATH = 'shelf_life_prediction_model.keras'
    SCALER_PATH = 'scaler.pkl'

    # Load the trained model and scaler globally
    model = None
    scaler = None

    print("Attempting to load model and scaler...") # Add this print statement

    try:
        # Check if files exist before attempting to load
        model_exists = os.path.exists(MODEL_PATH)
        scaler_exists = os.path.exists(SCALER_PATH)

        print(f"Model file exists: {model_exists}") # Add this print statement
        print(f"Scaler file exists: {scaler_exists}") # Add this print statement


        if model_exists and scaler_exists:
            model = tf.keras.models.load_model(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("Model and scaler loaded successfully.")
        else:
            print(f"Error loading model or scaler: File not found. Check paths: {MODEL_PATH}, {SCALER_PATH}")

    except Exception as e:
        print(f"Error loading model or scaler: An unexpected error occurred: {e}")
        # model and scaler remain None

    print("Model and scaler loading block finished.") # Add this print statement

@app.route('/')
def index():
    # Render the input form page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if model and scaler are loaded before processing the request
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded. Please check server logs for details."}), 500

    try:
        # Get the input values from the form
        moisture = float(request.form.get('moisture'))
        storage_temp = float(request.form.get('storage_temp'))
        rh = float(request.form.get('rh'))
        initial_ffa = float(request.form.get('initial_ffa'))
        initial_microbial_count = float(request.form.get('initial_microbial_count'))

        # Prepare the input data for scaling and prediction
        # Ensure the order matches the training data features
        input_data = np.array([[moisture, storage_temp, rh, initial_ffa, initial_microbial_count]])

        # Scale the input data
        scaled_input_data = scaler.transform(input_data)

        # Make the prediction
        prediction = model.predict(scaled_input_data)
        predicted_shelf_life = float(prediction[0][0]) # Extract the scalar value and cast to float

        # You can choose to render a results page or return JSON
        # For an API endpoint, returning JSON is typical:
        return jsonify({"predicted_shelf_life": predicted_shelf_life})

        # If you want to render a results page, uncomment the following lines:
        # return render_template('results.html', prediction=predicted_shelf_life)


    except (ValueError, TypeError):
        # Catch both ValueError (for float conversion) and TypeError (if .get() returns None)
        return jsonify({"error": "Invalid input. Please ensure all fields are filled with numeric values."}), 400
    except Exception as e:
        # Catch any other unexpected errors during the prediction process
        return jsonify({"error": f"An error occurred during prediction: {e}"}), 500

if __name__ == '__main__':
    # This is used for running the app locally for testing.
    # In a production environment (like Render), a WSGI server like Gunicorn is used.
    app.run(debug=True)
