#Let's define the model now
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Scripts.processed_data import categories
from Scripts.extract_features import X_train, y_train, X_test, y_test

class NNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
class DocumentsDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index].toarray().squeeze(), self.labels[index]

    def __len__(self):
        return len(self.labels)

# Set hyperparameters
input_dim = X_train.shape[1]
hidden_dim = 100
output_dim = len(categories)
learning_rate = 0.001
num_epochs = 10
batch_size = 32
    
model = NNetwork(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_dataset = DocumentsDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = DocumentsDataset(X_test, y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)