import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from tqdm import tqdm
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ------------------ Data Preparation ------------------

data = pd.read_csv('input/auto-mpg[1].csv')
data.replace('?',np.nan, inplace= True)
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
data['horsepower'].fillna(data['horsepower'].median(), inplace = True)
data = data.drop(['car name'], axis=1)
target = data['mpg']
features = data.drop(['mpg'], axis=1)

from sklearn.model_selection import train_test_split

# Convert to PyTorch tensors
X_tensor = torch.tensor(features.values, dtype=torch.float32).to(device)
y_tensor = torch.tensor(target.values, dtype=torch.float32).to(device)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
print(f'Train shape: {X_train.shape}, {y_train.shape}')
print(f'Test shape: {X_test.shape}, {y_test.shape}')

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=64, 
                                          shuffle=False)

# ------------------ Model Definition ------------------

num_epochs = 300

class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ------------------ Training ------------------

net = NN(7,1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)


for epoch in range(num_epochs):
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        targets = targets.to(device)
        
        data = data.reshape(data.shape[0], -1)

        predict = net(data)
        loss = criterion(predict, targets)

        # Backward pass: compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # Optimization step: update the model parameters
        optimizer.step()

def check_accuracy(loader, model):
    test_loss = 0
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], -1)

            # Forward pass: compute the model output
            y_pred = model(x.to(device))
            loss = criterion(y_pred, y)
            test_loss += loss.item()
        test_loss = test_loss / len(test_loader)
        print(f"Loss: {test_loss}")
    
    model.train()  # Set the model back to training mode

# Final accuracy check on training and test sets
check_accuracy(train_loader, net)
check_accuracy(test_loader, net)