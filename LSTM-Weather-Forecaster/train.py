import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

from preprocess import load_and_clean_data, scale_data, create_sliding_windows

def build_lstm_model(input_shape):
    """Defines and compiles the LSTM forecasting architecture."""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    print("Loading data...")
    df = load_and_clean_data('cleaned_weather.csv')
    
    print("Scaling and preparing sequences...")
    scaled_features, scaled_target, scaler_x, scaler_y = scale_data(df)
    X, y = create_sliding_windows(scaled_features, scaled_target)
    
    # Simple Train-Test split
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    print("Building model...")
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary()
    
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=256,
        verbose=1
    )
    
    print("Evaluating model...")
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_original = scaler_y.inverse_transform(y_test)
    
    # Calculate metrics
    r2 = r2_score(y_test_original, y_pred)
    mae = mean_absolute_error(y_test_original, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
    
    print("\n--- Evaluation Metrics ---")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE:      {mae:.4f}")
    print(f"RMSE:     {rmse:.4f}")
    
    print("Saving model...")
    model.save('weather_model.keras')
    print("Model saved to 'weather_model.keras'.")

if __name__ == '__main__':
    main()
