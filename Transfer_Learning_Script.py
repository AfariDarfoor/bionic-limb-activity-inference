import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
'''Portions of this code were adopted from the github repository detailing how to use the publicly available pre-trained Harnet10 model
Please see : https://github.com/OxWearables/ssl-wearables'''

def load_and_preprocess_data(train_path: str, test_path: str, sequence_length: int = 300):
    """Loads CSV data, encodes categorical labels consistently, and slices sequences."""
    train_df = pd.read_csv(train_path).dropna()
    test_df = pd.read_csv(test_path).dropna()

    #Label Encoding across both splits
    label_encoder = LabelEncoder()
    train_df['Activity_Code'] = label_encoder.fit_transform(train_df['Activity'])
    test_df['Activity_Code'] = label_encoder.transform(test_df['Activity'])

    def create_sequences(df, seq_len):
        X_list, y_list = [], []
        # Group by Activity to avoid boundary bleeding across distinct activities
        for _, group in df.groupby('Activity_Code'):
            features = group[['X', 'Y', 'Z']].values
            label = group['Activity_Code'].iloc[0]
            
            num_seqs = len(features) // seq_len
            if num_seqs == 0:
                continue
                
            features_trimmed = features[:num_seqs * seq_len]
            reshaped_features = features_trimmed.reshape(-1, seq_len, 3).transpose(0, 2, 1)
            
            X_list.append(reshaped_features)
            y_list.append(np.full(num_seqs, label))

        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        return torch.FloatTensor(X), torch.LongTensor(y)

    X_train, y_train = create_sequences(train_df, sequence_length)
    X_test, y_test = create_sequences(test_df, sequence_length)
    
    num_classes = len(label_encoder.classes_)
    return X_train, y_train, X_test, y_test, num_classes


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=20):
    """Trains the HAR network."""
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")


def evaluate_model(model, test_loader, device):
    """Evaluates classification accuracy on the test dataset."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Set Accuracy: {accuracy:.2f}%")
    return accuracy


def main():
    # Parameters
    train_path = 'path/to/your/training_sensor_data/.csv'
    test_path = 'path/to/your/testing_sensor_data/.csv'
    sequence_length = 300
    batch_size = 8
    learning_rate = 0.0005  
    num_epochs = 20

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load & Preprocess
    X_train, y_train, X_test, y_test, num_classes = load_and_preprocess_data(
        train_path, test_path, sequence_length
    )

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # Load Pre-trained HARNet Model via PyTorch Hub
    repo = 'OxWearables/ssl-wearables'
    print("Loading pre-trained HARNet10 model...")
    harnet10 = torch.hub.load(repo, 'harnet10', class_num=num_classes, pretrained=True)
    harnet10 = harnet10.to(device)

    # Optimization Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(harnet10.parameters(), lr=learning_rate)

    # Execute Pipeline
    print("Starting Training...")
    train_model(harnet10, train_loader, criterion, optimizer, device, num_epochs=num_epochs)

    print("Evaluating Model...")
    evaluate_model(harnet10, test_loader, device)


if __name__ == "__main__":
    main()