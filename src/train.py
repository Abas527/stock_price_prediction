import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from data_preprocessing import scaling

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_lookback(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:

    for i in range(1, lookback + 1):
        df[f"Close_lag_{i}"] = df["Close"].shift(i)
    df = df.dropna().reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(["Close"], axis=1) 
    y = df["Close"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    return x_train, x_test, y_train, y_test


def convert_to_tensor(x: pd.DataFrame, y: pd.Series) -> Tuple[torch.Tensor, torch.Tensor]:

    lag_cols = sorted([col for col in x.columns if col.startswith("Close_lag_")],
                      key=lambda s: int(s.split("_")[-1]))
    seq_len = len(lag_cols)
    # shape: (samples, seq_len, features=1)
    x_arr = x[lag_cols].values.astype(np.float32)
    x_tensor = torch.from_numpy(x_arr).view(-1, seq_len, 1)
    y_arr = y.values.astype(np.float32).reshape(-1, 1)
    y_tensor = torch.from_numpy(y_arr)
    return x_tensor, y_tensor


def create_dataloader(x_tensor: torch.Tensor, y_tensor: torch.Tensor, batch_size: int = 32, shuffle: bool = True):
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class LSTMModel(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 50, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        
        out, (h_n, c_n) = self.lstm(x)  # out: (batch, seq_len, hidden_size)
        out = self.dropout(out)
        
        out = out[:, -1, :]  # (batch, hidden_size)
        out = self.fc(out)   # (batch, 1)
        return out


def train_model(model: nn.Module, train_loader: DataLoader, criterion, optimizer, num_epochs: int = 100):
    model.to(DEVICE)
    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_losses = []
        for inputs, targets in train_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)           # (batch, 1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{num_epochs}]  avg loss: {np.mean(epoch_losses):.6f}")
    return model




def evaluate_model(model: nn.Module, x_test_tensor: torch.Tensor, y_test_tensor: torch.Tensor):
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        x_test_tensor = x_test_tensor.to(DEVICE)
        y_test_tensor = y_test_tensor.to(DEVICE)
        preds = model(x_test_tensor)                # (samples, 1)
        mse = nn.MSELoss()(preds, y_test_tensor)
    print(f"Test MSE: {mse.item():.6f}")
    
    return preds.cpu().numpy(), y_test_tensor.cpu().numpy()


def main():
    df = pd.read_csv("data/Apple_stock_price_preprocessed.csv")
    lookback = 30
    df = apply_lookback(df, lookback=lookback)

    x_train, x_test, y_train, y_test = split_data(df)
    x_train_tensor, y_train_tensor = convert_to_tensor(x_train, y_train)
    x_test_tensor, y_test_tensor = convert_to_tensor(x_test, y_test)

    train_loader = create_dataloader(x_train_tensor, y_train_tensor, batch_size=32, shuffle=True)

    model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, dropout=0.2)
    model.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model = train_model(model, train_loader, criterion, optimizer, num_epochs=500)

    preds, y_true = evaluate_model(model, x_test_tensor, y_test_tensor)
    
    preds=preds.flatten()
    y_true=y_true.flatten()

    mean,std=scaling()
    preds=preds*std+mean
    y_true=y_true*std+mean


    print(type(preds), preds.shape)
    for i in range(min(10, len(preds))):
        print(f"pred: {preds[i]:.6f}    true: {y_true[i]:.6f}")
    
    torch.save(model.state_dict(), "model/lstm_model.pth")


if __name__ == "__main__":
    main()
