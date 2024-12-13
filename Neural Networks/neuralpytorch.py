import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def load_data(file_path, input_cols, target_col):
    df = pd.read_csv(file_path, header=None)
    inputs = df.iloc[:, input_cols].values
    targets = df.iloc[:, target_col].values
    return inputs, targets

train_X, train_y = load_data('bank-note/train.csv', slice(0, 4), 4)
test_X, test_y = load_data('bank-note/test.csv', slice(0, 4), 4)

def prepare_dataloader(inputs, targets, batch_size, shuffle):
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, depth, width, activation):
        super(NeuralNetwork, self).__init__()
        layers = []
        prev_layer_size = input_size
        activation_fn = nn.Tanh() if activation == 'tanh' else nn.ReLU()
        initializer = nn.init.xavier_uniform_ if activation == 'tanh' else nn.init.kaiming_uniform_

        for _ in range(depth):
            layer = nn.Linear(prev_layer_size, width)
            initializer(layer.weight)
            layers.extend([layer, activation_fn])
            prev_layer_size = width

        layers.append(nn.Linear(prev_layer_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")


def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()
    return correct / total


input_size = train_X.shape[1]
output_size = 1
batch_size = 32
learning_rate = 1e-3
depths = [3, 5, 9]
widths = [5, 10, 25, 50, 100]
activations = ['tanh', 'relu']


train_loader = prepare_dataloader(train_X, train_y, batch_size, shuffle=True)
test_loader = prepare_dataloader(test_X, test_y, batch_size, shuffle=False)

for depth in depths:
    for width in widths:
        for activation in activations:
            print(f"Training with depth={depth}, width={width}, activation={activation}")
            model = NeuralNetwork(input_size, output_size, depth, width, activation)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            train_model(model, train_loader, criterion, optimizer)
            train_accuracy = evaluate_model(model, train_loader)
            test_accuracy = evaluate_model(model, test_loader)

            print(f"Train error: {1 - train_accuracy:.4f}")
            print(f"Test error: {1 - test_accuracy:.4f}")
