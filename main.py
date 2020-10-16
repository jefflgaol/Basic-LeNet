import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import os

# Hyperparameter
EPOCH = 1
BATCH_SIZE = 32
LR = 0.001

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

# Dataset transformation
transform_compose = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

# Train data
train_data = torchvision.datasets.MNIST(
    root='dataset/',
    train=True,                                     
    transform=transform_compose, # diubah ke dalam format c x h x w dan dinormalisasi    
    download=True)
train_loader = Data.DataLoader(
    dataset=train_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True)
# print(train_data.train_data.size())
# print(train_data.train_labels.size())

# Validation data
valid_data = torchvision.datasets.MNIST(
    root='dataset/',
    train=False,
    transform=transform_compose)
valid_loader = Data.DataLoader(
    dataset=valid_data,
    batch_size=BATCH_SIZE,
    shuffle=True)

# LeNet
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),      # 6 x 28 x 28
            nn.ReLU(),                                                              # dimodifikasi dari tanh, sebelum bagian ini bisa gunakan batch norm
            nn.MaxPool2d(kernel_size=2),                                            # 6 x 14 x 14
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),     # 16 x 10 x 10
            nn.ReLU(),                                                              # dimodifikasi from tanh
            nn.MaxPool2d(kernel_size=2),                                            # 16 x 5 x 5
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),   # 120 x 1 x 1
            nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),                            # 84
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10))                             # 10
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.cnn(x)
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = self.softmax(logits)
        return logits, probs

model = LeNet().to(DEVICE)
# print(model) # Menampilkan arsitektur secara detail
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    model.train()
    running_loss = 0
    for step, (x, y_true) in enumerate(train_loader):
        # Training
        x = x.to(DEVICE)
        y_true = y_true.to(DEVICE)              # [BATCH_SIZE]
        y_pred, _ = model(x)                    # Output dari LeNet [BATCH_SIZE][ONE_HOT]
        loss =  criterion(y_pred, y_true)       # Menghitung cross entropy loss
        optimizer.zero_grad()                   # Zeroing gradient
        loss.backward()                         # Menghitung gradient
        running_loss += loss.item() * x.size(0) # Dikali dengan batch size
        optimizer.step()                        # Melangkah di arah yang dituju gradient
        # Validation
        if step % 100 == 0:
            with torch.no_grad(): # Mematikan autograd engine, mengurangi penggunaan memory
                model.eval() # Melakukan notifikasi semua layer untuk menggunakan mode evaluasi
                running_loss_eval = 0
                for x, y_true in valid_loader:
                    x = x.to(DEVICE)
                    y_true = y_true.to(DEVICE)
                    y_pred, _ = model(x)
                    loss_eval = criterion(y_pred, y_true)
                    running_loss_eval += loss_eval.item() * x.size(0)
                epoch_loss_eval = running_loss_eval / len(valid_loader.dataset)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(epoch_loss)


# Data Visualization
cols = 5
rows = 5
fig = plt.figure()
for index in range(1, cols * rows + 1):
    plt.subplot(rows, cols, index)
    plt.axis('off')
    plt.imshow(valid_data.data[index], cmap='gray_r')
    with torch.no_grad():
        model.eval()
        _, probs = model(valid_data[index][0].unsqueeze(0).to(DEVICE))
    title = '%d: %.2f%%' % (torch.argmax(probs), torch.max(probs * 100))
    plt.title(title, fontsize=10)
plt.show()

# Saving pytorch model
torch.save(model, 'model.pkl')
# torch.save(model.state_dict(), 'params.pkl')

# Restore
loaded_model = torch.load("model.pkl")
with torch.no_grad():
    model.eval()
    _, probs = loaded_model(valid_data[0][0].unsqueeze(0).to(DEVICE))
    plt.imshow(valid_data.data[0], cmap='gray_r')
    plt.show()
    print("Result:", torch.argmax(probs).item())