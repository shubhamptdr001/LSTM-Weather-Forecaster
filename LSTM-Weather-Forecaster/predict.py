import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

def load_inference_assets(model_path='weather_model.keras', scaler_x_path='scaler_x.pkl', scaler_y_path='scaler_y.pkl'):
    """Loads trained model and preprocessing scalers."""
    model = tf.keras.models.load_model(model_path)
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    return model, scaler_x, scaler_y

def predict_future_temp(csv_file_path):
    """Predicts temperature 24 hours in the future using the last 24 hours of data."""
    df = pd.read_csv(csv_file_path)
    features = ['T', 'rh', 'p', 'wv', 'SWDR']
    
    if len(df) < 144:
        raise ValueError("Input CSV file must contain at least 144 observations (24 hours at 10-minute intervals).")
        
    last_window = df[features].tail(144).values
    model, scaler_x, scaler_y = load_inference_assets()
    
    # Scale and reshape input
    scaled_window = scaler_x.transform(last_window)
    input_data = scaled_window.reshape(1, 144, len(features))
    
    # Inference
    prediction_scaled = model.predict(input_data)
    predicted_temp = scaler_y.inverse_transform(prediction_scaled)[0][0]
    
    return predicted_temp

if __name__ == '__main__':
    # Example usage:
    try:
        pred = predict_future_temp('cleaned_weather.csv')
        print(f"Predicted temperature 24 hours in the future: {pred:.2f}°C")
    except Exception as e:
        print(f"Error making prediction: {e}")
