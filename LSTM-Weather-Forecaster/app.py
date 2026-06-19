import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Weather Predictor", layout="wide")
st.title("🌡️ 24-Hour Temperature Forecast")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('weather_model.keras')
    scaler_x = joblib.load('scaler_x.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    return model, scaler_x, scaler_y

try:
    model, scaler_x, scaler_y = load_assets()
except:
    st.error("Model files not found! Please run the training script and save 'weather_model.h5', 'scaler_x.pkl', and 'scaler_y.pkl'.")
    st.stop()

st.sidebar.header("Upload Latest Data")
uploaded_file = st.sidebar.file_uploader("Upload your previous 24 hours dataset.", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    st.subheader("Recent Weather Data (Last 24 Hours)")
    features = ['T', 'rh', 'p', 'wv', 'SWDR']
    st.dataframe(df[features].tail(10))

    if len(df) >= 144:
        #Prepare the window
        last_window = df[features].tail(144).values
        last_window_scaled = scaler_x.transform(last_window)
        input_data = last_window_scaled.reshape(1, 144, 5)

        # 4.Make Prediction
        if st.button('Predict Tomorrow\'s Temperature'):
            with st.spinner('Calculating...'):
                pred_scaled = model.predict(input_data)
                prediction = scaler_y.inverse_transform(pred_scaled)
                
                #Show Result
                st.success(f"### Predicted Temperature After 24 hours: {prediction[0][0]:.2f}°C")
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(df['T'].tail(144).values, label="Past 24h")
                #Plot the prediction point
                ax.scatter(144 + 144, prediction[0][0], color='red', label="Predicted Temperature After a day")
                ax.set_ylabel("Temperature (°C)")
                ax.grid(axis='y',linestyle='--',alpha=0.6,color='gray')
                ax.legend()
                st.pyplot(fig)
    else:
        st.warning("Need at least 144 rows of data (24 hours) to make a prediction.")

else:
    st.info("Waiting for CSV file upload to begin forecasting.")
