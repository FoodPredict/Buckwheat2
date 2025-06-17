import tensorflow as tf
import joblib
import pandas as pd

# Load the trained model
model = tf.keras.models.load_model('shelf_life_prediction_model.keras')

# Load the fitted scaler
scaler = joblib.load('scaler.pkl')

def predict_shelf_life(moisture, storage_temp, rh, initial_ffa, initial_microbial_count):
    """
    Predicts the shelf life of buckwheat flour.

    Args:
        moisture (float): Moisture content in percentage.
        storage_temp (float): Storage temperature in degrees Celsius.
        rh (float): Relative humidity in percentage.
        initial_ffa (float): Initial free fatty acid percentage.
        initial_microbial_count (float): Initial microbial count in log CFU/g.

    Returns:
        float: Predicted shelf life in days.
    """
    # Create a pandas DataFrame from the input data
    # The column order must match the order used during training
    input_data = pd.DataFrame([[moisture, storage_temp, rh, initial_ffa, initial_microbial_count]],
                              columns=['Moisture (%)', 'Storage Temp (°C)', 'RH (%)', 'Initial FFA (%)', 'Initial Microbial Count (log CFU/g)'])

    # Scale the input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data)

    # Make a prediction using the loaded model
    prediction = model.predict(input_data_scaled)

    # The prediction is likely a 2D array [[value]], so extract the single value
    return prediction[0][0]

if __name__ == '__main__':
    # Example usage (for testing purposes)
    # You would get these values from your web application input
    example_moisture = 11.5
    example_storage_temp = 25.0
    example_rh = 65.0
    example_initial_ffa = 0.35
    example_initial_microbial_count = 2.2

    predicted_days = predict_shelf_life(example_moisture, example_storage_temp, example_rh, example_initial_ffa, example_initial_microbial_count)
    print(f"Predicted Shelf Life: {predicted_days:.2f} days")