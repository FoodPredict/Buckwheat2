from flask import Flask, render_template, request
import joblib
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = tf.keras.models.load_model('shelf_life_prediction_model.keras')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None # Set to None if loading fails
    scaler = None

@app.route('/')
def index():
    # Render the input form page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return "Error: Model or scaler not loaded.", 500

    try:
        # Get the input values from the form
        moisture = float(request.form['moisture'])
        storage_temp = float(request.form['storage_temp'])
        rh = float(request.form['rh'])
        initial_ffa = float(request.form['initial_ffa'])
        initial_microbial_count = float(request.form['initial_microbial_count'])

        # Prepare the input data for scaling and prediction
        # Ensure the order matches the training data features
        input_data = np.array([[moisture, storage_temp, rh, initial_ffa, initial_microbial_count]])

        # Scale the input data
        scaled_input_data = scaler.transform(input_data)

        # Make the prediction
        prediction = model.predict(scaled_input_data)
        predicted_shelf_life = prediction[0][0] # Extract the scalar value

        # Render the results page with the prediction
        return render_template('results.html', prediction=predicted_shelf_life)

    except ValueError:
        return "Invalid input. Please enter numeric values.", 400
    except Exception as e:
        return f"An error occurred during prediction: {e}", 500

if __name__ == '__main__':
    # Run the Flask app
    # In a production environment, you would use a production-ready server
    app.run(debug=True) # debug=True is useful for development