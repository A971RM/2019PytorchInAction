import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
txform = transforms.Compose(transforms=[
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])
train_set = datasets.MNIST(root='./data', train=True, transform=txform, download=True)
ttest_set = datasets.MNIST(root='./data', train=False, transform=txform, download=True)

batch_size = 200
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
ttest_loader = DataLoader(dataset=ttest_set, batch_size=batch_size, shuffle=False)

epochs = 10
learning_rate = 0.01

w1, b1 = torch.randn(200, 784, device=device, requires_grad=True), \
         torch.randn(200, device=device, requires_grad=True)
w2, b2 = torch.randn(200, 200, device=device, requires_grad=True), \
         torch.randn(200, device=device, requires_grad=True)
w3, b3 = torch.randn(10, 200, device=device, requires_grad=True), \
         torch.randn(10, device=device, requires_grad=True)

torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)


def forward(x):
    x = x @ w1.t() + b1
    x = F.relu(x)
    x = x @ w2.t() + b2
    x = F.relu(x)
    x = x @ w3.t() + b3
    return x


optimizer = optim.SGD([w1, b1, w2, b2, w3, b3], lr=learning_rate)
criteon = nn.CrossEntropyLoss()

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(data.size(0), -1).to(device)
        target = target.to(device)
        logits = forward(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_loss = 0.
    correct = 0.

    for tdata, ttarget in ttest_loader:
        tdata = tdata.view(tdata.size(0), -1).to(device)
        ttarget = ttarget.to(device)
        logits = forward(tdata)
        loss = criteon(logits, ttarget)
        test_loss += loss.item()
        pred = torch.argmax(logits, dim=1)
        correct += pred.eq(ttarget).sum()
    test_loss /= len(ttest_loader.dataset)
    print("Test Loss: {:.4f}, Accury:{:.4f}".format(test_loss, 100. * correct / len(ttest_loader.dataset)))
