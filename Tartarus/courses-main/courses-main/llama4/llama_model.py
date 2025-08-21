import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from lesson_4_llama4_feedforward_code import SimplifiedLlama4FFN

# Sample dataset class
class SampleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Define the Llama model (simplified for demonstration)
class LlamaModel(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act, ffn_bias, rms_norm_eps):
        super(LlamaModel, self).__init__()
        self.ffn = SimplifiedLlama4FFN({
            'hidden_size': hidden_size,
            'intermediate_size': intermediate_size,
            'hidden_act': hidden_act,
            'ffn_bias': ffn_bias,
            'rms_norm_eps': rms_norm_eps,
        })

    def forward(self, x):
        return self.ffn(x)

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

if __name__ == "__main__":
    # Configuration
    hidden_size = 128
    intermediate_size = 256
    hidden_act = "silu"
    ffn_bias = False
    rms_norm_eps = 1e-5
    num_epochs = 10
    batch_size = 2

    # Sample data
    data = torch.randn(100, 10, hidden_size)  # 100 samples, sequence length of 10
    labels = torch.randn(100, 10, hidden_size)  # Dummy labels

    dataset = SampleDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, criterion, and optimizer
    model = LlamaModel(hidden_size, intermediate_size, hidden_act, ffn_bias, rms_norm_eps)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, num_epochs)

    # Save the trained model
    save_model(model, os.path.join(os.path.dirname(__file__), 'llama_model.pth'))