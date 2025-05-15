import torch
from torch import nn
import snntorch as snn


def rate_encode(
    tensor: torch.Tensor,
    num_timesteps: int = 50,
    gain: float = 1,
):
    from snntorch import spikegen

    return spikegen.rate(data=tensor * gain, num_steps=num_timesteps)


class neuron_layer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        alpha: float = 0.9,
        beta: float = 0.9,
    ):
        super().__init__()

        self.ws = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
        )
        self.lif = snn.Synaptic(
            alpha=alpha,
            beta=beta,
        )
        self.reset_params()

    def reset_params(self):
        self.syn, self.mem = self.lif.reset_mem()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        current = self.ws(x)
        spikes, self.syn, self.mem = self.lif(current, self.syn, self.mem)
        return spikes


class SNN(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_hidden: int,
        num_outputs: int,
    ) -> None:
        super().__init__()

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.l1 = neuron_layer(in_features=num_inputs, out_features=num_hidden)
        self.l2 = neuron_layer(in_features=num_hidden, out_features=num_outputs)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x.flatten(start_dim=2)

        self.l1.reset_params()
        self.l2.reset_params()

        num_timesteps = x.size(0)
        batch_size = x.size(1)

        out_spikes = torch.zeros(batch_size, self.num_outputs)

        for timestep in range(num_timesteps):
            foo = self.l1(x[timestep])
            baz = self.l2(foo)
            out_spikes += baz

        row_sums = out_spikes.sum(dim=1, keepdim=True)
        safe_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        probabilities = out_spikes / safe_sums

        return probabilities


from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batch_size = 128
data_path = "data/mnist"

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define a transform
transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
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


num_inputs = 28 * 28
num_hidden = 32
num_outputs = 10

snn = SNN(
    num_inputs=num_inputs,
    num_hidden=num_hidden,
    num_outputs=num_outputs,
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=1e-3)

num_epochs = 1
num_timesteps = 50

for epoch in range(num_epochs):
    print(f"E {epoch+1}")

    snn.train()

    for index, (data, targets) in enumerate(train_dataloader):
        print_loss = False
        if (index + 1) % 10 == 0:
            print_loss = True
            print(f"\tBatch {index+1:,}/{len(train_dataloader):,}")

        optimizer.zero_grad()
        # print(data.size())
        # print(targets.size())
        data = rate_encode(data, num_timesteps=num_timesteps)
        probabilities = snn(data)
        # print(probabilities.size())
        # print(targets.size())
        loss = loss_fn(probabilities, targets)
        loss.backward()
        optimizer.step()

        if print_loss == True:
            print(f"\tLoss: {loss.item()}")


snn.eval()

with torch.no_grad():
    correct, total = 0, 0
    for index, (data, targets) in enumerate(train_dataloader):
        if (index + 1) % 10 == 0:
            print(f"\tBatch {index+1:,}/{len(train_dataloader):,}")
        data = rate_encode(data, num_timesteps=num_timesteps)

        probabilities = snn(data)
        choices = probabilities.argmax(dim=1)

        matches = (choices == targets).sum().item()

        total += targets.size(0)
        correct += matches

    print(f"Correct/Total: {correct:,}/{total:,} = {(correct/total)*100:.3f}%")

# Correct/Total: 51,748/59,904 = 86.385%