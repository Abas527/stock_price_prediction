import streamlit as st
import pandas as pd
import numpy as np
import torch

from train import LSTMModel
from data_preprocessing import scaling
from predict import predict


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df['Date']=pd.to_datetime(df['Date'],utc=True)
    df.drop(['Dividends','Stock Splits'],axis=1,inplace=True)

    day=df['Date'].dt.day
    month=df['Date'].dt.month
    year=df['Date'].dt.year

    df['date']=pd.to_datetime(dict(year=year,month=month,day=day))

    df.drop(['Date'],axis=1,inplace=True)
    df.set_index('date', inplace=True)
    return df



def main():
    st.title("Apple Stock Price Prediction")
    
    df = pd.read_csv("data/Apple_stock_price.csv")
    df = preprocessing(df)
    st.subheader("Historical Stock Prices")
    st.write(df.tail())

    input_lookback = st.slider("Select Lookback Period", min_value=10, max_value=100, value=30, step=5)

    lookback = input_lookback
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, dropout=0.2)
    model.load_state_dict(torch.load("model/lstm_model.pth", map_location=torch.device('cpu')))
    model.eval()

    last_sequence = df['Close'].values[-lookback:]
    last_sequence = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(-1)  # (lookback, 1)

    predicted_price = predict(model, last_sequence, lookback)
    mean, std = scaling(lookback=lookback)
    predicted_price = predicted_price * std + mean

    st.subheader("Predicted Next Closing Price")
    st.write(f"${predicted_price:.2f}")

    st.subheader("Price Trend Visualization")
    
    st.line_chart(df['Close'])
    st.line_chart(df['Volume'])


    st.write("Developed by Anil khatiwada")
    st.write("GitHub: [Anil-khatiwada](https://github.com/Abas527)")
    

if __name__ == "__main__":
    main()