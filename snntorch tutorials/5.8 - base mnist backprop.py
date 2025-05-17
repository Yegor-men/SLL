import torch
from torch import nn


class Model(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()

        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=out_features),
        )

    def forward(self, x):
        return self.layer(x)


from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batch_size = 1
data_path = "data/mnist"

# Define a transform
transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,)),
    ]
)

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

train_dataloader = DataLoader(
    mnist_train, batch_size=batch_size, shuffle=True, drop_last=True
)
test_dataloader = DataLoader(
    mnist_test, batch_size=batch_size, shuffle=True, drop_last=True
)

model = Model(in_features=28 * 28, out_features=10)

epochs = 10

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

import matplotlib.pyplot as plt

correct_or_not = []
averages = []

for epoch in range(epochs):
    print(f"{epoch}")
    
    for index, (data, targets) in enumerate(train_dataloader):
        optimizer.zero_grad()
        
        if (index + 1) % 1000 == 0:
            print(f"{index+1:,}/{len(train_dataloader):,}")

            ao1000 = sum(correct_or_not[-1000:]) / 1000
            averages.append(ao1000)

            plt.figure("Live graph")
            plt.clf()
            plt.plot(averages)
            plt.ylim(0, 1)
            plt.show(block=False)
            plt.pause(0.00001)
        
        distribution = model(data)
        
        choices = distribution.argmax(dim=1)
        correct = targets
        
        loss = loss_fn(distribution, targets)
        loss.backward()
        optimizer.step()

        num_matches = (choices == correct).sum().item()
        correct_or_not.append(num_matches)