import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.regressor = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.regressor(last_out).squeeze(-1)


def prepare_sequences(
    X: torch.Tensor,
    y: torch.Tensor,
    seq_length: int = 24,
) -> tuple[torch.Tensor, torch.Tensor]:
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i: i + seq_length])
        y_seq.append(y[i + seq_length])
    return torch.stack(X_seq), torch.tensor(y_seq)
