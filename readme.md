# Stock Price Prediction with LSTM

This project predicts Apple stock prices using an LSTM neural network. It covers data gathering, preprocessing, model training, and prediction, all organized for easy experimentation and deployment.

## Project Structure

```
stock_price_prediction/
│
├── data/
│   ├── Apple_stock_price.csv
│   └── Apple_stock_price_preprocessed.csv
│
├── model/
│   └── lstm_model.pth
│
├── src/
│   ├── app.py
│   ├── data_gathering.py
│   ├── data_preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── venv/
│   └── ... (Python virtual environment)
│
└── readme.md
```

## Features

- **Data Gathering:** Download and update Apple stock price data.
- **Preprocessing:** Clean and transform data for LSTM input.
- **Model Training:** Train an LSTM model to predict future prices.
- **Prediction:** Use the trained model to forecast prices.
- **Web App:** Simple interface for predictions (`app.py`).

## Getting Started

### 1. Clone the Repository

```sh
git clone <repo-url>
cd stock_price_prediction
```

### 2. Set Up Environment

```sh
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Data Preparation

- Place raw data in Apple_stock_price.csv.
- Run preprocessing:

```sh
python src/data_preprocessing.py
```

### 4. Train the Model

```sh
python src/train.py
```

- The trained model is saved as lstm_model.pth.

### 5. Make Predictions

```sh
python src/predict.py
```

### 6. Run the Web App

```sh
python src/app.py
```

## File Descriptions

- **data_gathering.py:** Fetches and updates stock data.
- **data_preprocessing.py:** Cleans and prepares data for modeling.
- **train.py:** Trains the LSTM model.
- **predict.py:** Loads the model and makes predictions.
- **app.py:** Flask web app for user interaction.

## Requirements

- Python 3.10+
- PyTorch
- Pandas
- NumPy
- Flask (for web app)

Install dependencies with:

```sh
pip install -r requirements.txt
```

## Model

- LSTM neural network for time series forecasting.
- Model weights saved in lstm_model.pth.

## Data

- Apple_stock_price.csv: Raw historical data.
- Apple_stock_price_preprocessed.csv: Cleaned and processed data.

## Usage Example

Train and predict from the command line:

```sh
python src/train.py
python src/predict.py
```

## License

This project is for educational purposes.

---

**Feel free to modify and extend for your own experiments!**