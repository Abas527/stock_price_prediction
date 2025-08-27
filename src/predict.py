import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from train import LSTMModel
from data_preprocessing import scaling


def predict(model: nn.Module, input_sequence: torch.Tensor, lookback: int) -> float:
    model.eval()
    with torch.no_grad():
        input_sequence = input_sequence.unsqueeze(0).to(next(model.parameters()).device) 
        pred = model(input_sequence)  # (1, 1)
    return pred.item()


def main():
    df = pd.read_csv("data/Apple_stock_price.csv")
    lookback = 30

    model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, dropout=0.2)
    model.load_state_dict(torch.load("model/lstm_model.pth", map_location=torch.device('cpu')))
    model.eval()

    last_sequence = df['Close'].values[-lookback:]
    last_sequence = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(-1)  # (lookback, 1)

    predicted_price = predict(model, last_sequence, lookback)
    mean, std = scaling(lookback=lookback)
    predicted_price = predicted_price * std + mean
    print(f"Predicted next closing price: {predicted_price:.2f}")

if __name__ == "__main__":
    main()