import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class LSTMModelDataset(Dataset):
    def __init__(self, data, seq_len, target_dim):
        self.data = data
        self.seq_len = seq_len
        self.target_dim = target_dim

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.seq_len]
        target = self.data[idx + self.seq_len, -self.target_dim:]
        return {
            'seq': torch.tensor(seq, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32)
        }

class LSTMModelConfig:
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, seq_len, batch_size, epochs):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.epochs = epochs

class LSTMModelTrainer:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

    def train(self, train_loader, val_loader):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                seq, target = batch['seq'].to(self.device), batch['target'].to(self.device)
                optimizer.zero_grad()
                output = self.model(seq)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch in val_loader:
                    seq, target = batch['seq'].to(self.device), batch['target'].to(self.device)
                    output = self.model(seq)
                    loss = criterion(output, target)
                    val_loss += loss.item()
                logger.info(f'Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}')

    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                seq = batch['seq'].to(self.device)
                output = self.model(seq)
                predictions.append(output.cpu().numpy())
        return np.array(predictions)

def init_lstm_model(config):
    model = LSTMModel(config.input_dim, config.hidden_dim, config.output_dim, config.num_layers, config.dropout)
    return model

def train_lstm_model(model, config, train_loader, val_loader):
    trainer = LSTMModelTrainer(model, config, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    trainer.train(train_loader, val_loader)

def predict_trajectory(model, config, test_loader):
    return trainer.predict(test_loader)

def load_data(file_path):
    data = pd.read_csv(file_path)
    scaler = MinMaxScaler()
    data[['x', 'y']] = scaler.fit_transform(data[['x', 'y']])
    return data

def create_dataset(data, seq_len, target_dim):
    dataset = LSTMModelDataset(data, seq_len, target_dim)
    return dataset

def create_data_loaders(dataset, batch_size):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

if __name__ == '__main__':
    config = LSTMModelConfig(input_dim=2, hidden_dim=64, output_dim=2, num_layers=2, dropout=0.2, seq_len=10, batch_size=32, epochs=100)
    data = load_data('data.csv')
    dataset = create_dataset(data, config.seq_len, config.output_dim)
    train_loader, val_loader = create_data_loaders(dataset, config.batch_size)
    model = init_lstm_model(config)
    train_lstm_model(model, config, train_loader, val_loader)
    predictions = predict_trajectory(model, config, train_loader)
    logger.info(predictions)