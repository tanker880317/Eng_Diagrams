import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def train_cnn():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the Symbols_pixel.csv dataset
    csv_path = "data/Symbols_pixel.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Dataset {csv_path} not found.")
        return

    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")

    # Standardize column headers
    num_cols = df.shape[1]
    new_cols = ['p_' + str(i) for i in range(1, num_cols)]
    new_cols.append('label')
    df.columns = new_cols[:num_cols]

    # Extract features and labels
    labels_raw = df['label'].values
    features_raw = df.iloc[:, 0:num_cols-1].values

    # Encode categorical labels to integers
    le = LabelEncoder()
    labels = le.fit_transform(labels_raw)
    num_classes = len(le.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Class mapping: {list(le.classes_)}")

    # Normalize pixels to [0, 1] and reshape to (N, 1, 100, 100)
    features = features_raw.astype(np.float32) / 255.0
    features = features.reshape(-1, 1, 100, 100)

    # 2. Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    print(f"Train size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}")

    # Convert to PyTorch Tensors
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 3. Define the CNN architecture (matching the EANN2020 paper layout)
    class SymbolCNN(nn.Module):
        def __init__(self, num_classes):
            super(SymbolCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # -> 50x50
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)  # -> 25x25
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 25 * 25, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = SymbolCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Train for 2 epochs to verify it runs successfully
    epochs = 2
    print(f"Training for {epochs} epochs to verify classification execution...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc*100:.2f}%")

    # 5. Evaluate on Validation set
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = outputs.max(1)
            val_total += y_batch.size(0)
            val_correct += predicted.eq(y_batch).sum().item()

    val_acc = val_correct / val_total
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print("==============================================================")
    print("[SUCCESS] CNN Classification model trained and verified successfully!")
    print("==============================================================")

if __name__ == "__main__":
    train_cnn()
