import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

x = torch.zeros(size=(1, 784), dtype=torch.float)
nn.init.kaiming_uniform_(x)
print(x.shape)

layer1 = nn.Linear(in_features=784, out_features=200)
layer2 = nn.Linear(in_features=200, out_features=200)
layer3 = nn.Linear(in_features=200, out_features=10)

for layer in [layer1, layer2, layer3]:
    print(layer)
    for param in layer.parameters():
        print(param.shape, param.type())
    x = layer(x)
    F.relu(x, inplace=True)
    print(x.shape)

txform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,), inplace=True)
])

train_set = datasets.MNIST(root='./data', train=True, transform=txform, download=True)
test_set = datasets.MNIST(root='./data', train=False, transform=txform, download=True)

batch_size = 200
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=784, out_features=200),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=200, out_features=200),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=200, out_features=10),
        )

    def forward(self, x):
        x = self.model(x)
        return x


net = MLP()
for param in net.parameters():
    if param.dim() > 1:
        nn.init.kaiming_normal_(param)
learning_rate = 0.01
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss()
epochs = 10

for epoch in range(epochs):
    for idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 ** 2)

        logits = net(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_loss = 0.
    correct = 0.
    for data, target in test_loader:
        data = data.view(-1, 28 ** 2)
        logits = net(data)
        test_loss += criteon(logits, target).item()
        pred = torch.argmax(logits, dim=1)
        correct += (pred == target).sum().item()
    print("epch {}, Loss {:4f}, Correct {:4f}".format(epoch, test_loss, correct * 100. / len(test_loader.dataset)))
