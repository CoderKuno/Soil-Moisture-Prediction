import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import firebase_admin
from firebase_admin import credentials, db
import time
import os

if not firebase_admin._apps:
    current_dir = os.path.dirname("C:/Users/Dave/OneDrive - usep.edu.ph/Documents/2023-2024 2nd Sem/Intelligent Information Systems/LE/Time Series Deployment")
    credentials_path = os.path.join(current_dir, 'credentials.json')
    cred= credentials.Certificate(credentials_path)
    firebase_admin.initialize_app(cred, {"databaseURL": "https://esp32-336d9-default-rtdb.asia-southeast1.firebasedatabase.app/"})

# Reference to your Firebase Realtime Database path
ref = db.reference('Sample/')
def load_data():
    df = pd.read_csv("C:/Users/Dave/OneDrive - usep.edu.ph/Documents/2023-2024 2nd Sem/Intelligent Information Systems/LE/data.csv")
    df= df.rename(columns={"ttime": "Date & Time",
                   "temp": "Temperature",
                   "humd": "Humidity",
                   "Soil_Moisture_Percentage": "Soil Moisture(%)"})
    
    df['Date & Time'] = pd.to_datetime(df['Date & Time'],format='%d/%m/%Y %H:%M')
    return df

# Function to fetch data from Firebase
def fetch_data():
    try:
        snapshot = ref.order_by_key().get()
        if snapshot:
            print("Data fetched successfully from Firebase.")
        else:
            print("No data found at the specified path.")
            
        data = []
        for timestamp, values in snapshot['readings'].items():
            timestamp_date = datetime.strptime(timestamp.strip(), '%d-%m-%Y %H:%M')
            if timestamp_date >= start_date:
                entry = {
                    'Date & Time': timestamp_date,
                    'Temperature': values.get('Temperature', None),
                    'Humidity': values.get('Humidity', None),
                    'Soil Moisture(%)': values.get('Moisture', None)
                }
                data.append(entry)
        
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print(f"Error fetching data from Firebase: {e}")
        return pd.DataFrame()

st.title("Soil Moisture Prediction App")
period = st.slider("Days of Forecast", 1, 7)


df = load_data()
# Display historical data
def table_hist(df):
    st.subheader("Historical Data (Last 100 Rows)")

    st.dataframe(df.tail(100).set_index('Date & Time'))

# Describe the dataset
    st.subheader("Data Description")
    st.write(df.describe())
# Plot the dataset
def plot_raw_data(df):
    fig = go.Figure()
    daily_df = df.resample('D', on='Date & Time').mean()
    fig.add_trace(go.Scatter(x=daily_df.index, y=daily_df["Soil Moisture(%)"]))
    fig.update_layout(
        title={
            'text': 'Soil Moisture Over Time',
            'font': {
                'size': 24
            }
        },
        xaxis_rangeslider_visible=True,
        yaxis=dict(
            title='Soil Moisture (%)'
        )
    )
    st.plotly_chart(fig)


# Function to update model
def update_model(df):
    df_train = df[["Date & Time", "Temperature", "Humidity", "Soil Moisture(%)"]]
    df_train = df_train.rename(columns={"Date & Time": "ds", "Soil Moisture(%)": "y"})
    m = Prophet()
    m.add_regressor('Temperature')
    m.add_regressor('Humidity')
    m.fit(df_train)
    return m

# Initial model training
model = update_model(df)

# Continuous learning loop
while True:   
    start_date= max(df['Date & Time'])
    new_df = fetch_data()

    # Concatenate the new DataFrame with the existing DataFrame
    df = pd.concat([df, new_df], ignore_index=True)
    table_hist(df)
    plot_raw_data(df)
    # Update model with the combined DataFrame
    model = update_model(df)

    # Forecasting
    future = model.make_future_dataframe(periods=period)
    future['Temperature'] = df['Temperature'].values[-1]  # Assuming temperature remains constant in the future
    future['Humidity'] = df['Humidity'].values[-1]  # Assuming humidity remains constant in the future
    forecast = model.predict(future)

    # Selecting only 'ds' and 'yhat' columns and renaming them
    forecast_filtered = forecast[['ds', 'yhat']]
    forecast_filtered = forecast_filtered.rename(columns={"ds": "Date & Time", "yhat": "Forecasted Soil Moisture(%)"})

    # Displaying forecasted data in a table
    st.subheader(f"Forecasted Soil Moisture for {period} Days")
    st.dataframe(forecast_filtered.tail(period).set_index(forecast_filtered.columns[0]))

    # Plotting the Forecasted data
    def plot_forecast_values(forecast):
        # Filter out historical data from the forecast DataFrame
        forecast_future = forecast[forecast['ds'] > max(df['Date & Time'])]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_future['ds'], y=forecast_future['yhat'], mode='lines', name='Forecast', line=dict(color='green')))
        fig.update_layout(title={'text': f'Forecasted Soil Moisture for {period} Days Plot','font': {'size': 24}},
                          xaxis_title='Date',
                          yaxis_title='Forecasted Value')
        st.plotly_chart(fig)

    plot_forecast_values(forecast)

    # Wait for a while before checking for updates again
    time.sleep(3600)  # 1 hour in seconds