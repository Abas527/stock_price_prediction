import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

st=StandardScaler()
mt=MinMaxScaler()


def scaling(lookback:int=50):
    df=pd.read_csv("data/Apple_stock_price.csv")
    df.sort_values(by='Date',inplace=True,ascending=False)

    close_values=np.array(df['Close'].tolist())
    close_values=close_values[::-1]
    close_values=[close_values[i] for i in range(len(close_values)-lookback,len(close_values))]

    return np.mean(close_values),np.std(close_values)
    


def main():
    df=pd.read_csv("data/Apple_stock_price.csv")

    df['Date']=pd.to_datetime(df['Date'],utc=True)
    df.drop(['Dividends','Stock Splits'],axis=1,inplace=True)

    day=df['Date'].dt.day
    month=df['Date'].dt.month
    year=df['Date'].dt.year

    df['date']=pd.to_datetime(dict(year=year,month=month,day=day))

    df.drop(['Date'],axis=1,inplace=True)

    df[['Open','High','Low','Close']]=st.fit_transform(df[['Open','High','Low','Close']])

    joblib.dump(st,"standard_scaler.save")
    df[['Volume']]=mt.fit_transform(df[['Volume']])
    df.sort_values(by='date',inplace=True)

    df.drop(['date'],axis=1,inplace=True)

    df.to_csv("data/Apple_stock_price_preprocessed.csv",index=False)
    
    print(df.head())


if __name__=="__main__":
    main()