import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_and_clean_data(file_path):
    """Loads and preprocesses raw weather data."""
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Select key weather features
    features = ['T', 'rh', 'p', 'wv', 'SWDR']
    df = df[features]
    
    # Handle missing values
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    return df

def scale_data(df, save_path_x='scaler_x.pkl', save_path_y='scaler_y.pkl'):
    """Fits MinMaxScaler and scales features and targets."""
    features = ['T', 'rh', 'p', 'wv', 'SWDR']
    
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    scaled_features = scaler_x.fit_transform(df[features])
    scaled_target = scaler_y.fit_transform(df[['T']])
    
    joblib.dump(scaler_x, save_path_x)
    joblib.dump(scaler_y, save_path_y)
    
    return scaled_features, scaled_target, scaler_x, scaler_y

def create_sliding_windows(scaled_features, scaled_target, sequence_length=144, forecast_horizon=144):
    """Creates input sequences (X) and corresponding future target values (y)."""
    X, y = [], []
    for i in range(len(scaled_features) - sequence_length - forecast_horizon + 1):
        X.append(scaled_features[i : i + sequence_length])
        y.append(scaled_target[i + sequence_length + forecast_horizon - 1])
        
    return np.array(X), np.array(y)

if __name__ == '__main__':
    df = load_and_clean_data('cleaned_weather.csv')
    scaled_features, scaled_target, _, _ = scale_data(df)
    X, y = create_sliding_windows(scaled_features, scaled_target)
    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
