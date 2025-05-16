import torch
from torch import nn
import snntorch as snn

torch.manual_seed(0)
torch.cuda.manual_seed(0)


class EPropLayer:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        alpha: float = 0.9,
        beta: float = 0.9,
        threshold: float = 1,
        eligibility_decay: float = 0.999,
        learning_rate: float = 0,
        batch_size: int = 1,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.decay_value = eligibility_decay
        self.effective_learning_rate = learning_rate * (1 - eligibility_decay)
        self.batch_size = batch_size

        self.weighted_sum = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
        )
        # nn.Linear has .weight, the weight matrix of size [out_features, in_features]
        # p.s. pytorch is goofy and decides to .T every time for some reason?? why??

        self.reset_states()

    def reset_states(self):
        self.eligibility_trace = torch.zeros_like(self.weighted_sum.weight)
        # eligibility trace is the same as weight: [out_features, in_features]

        self.syn = torch.zeros(self.batch_size, self.out_features)
        self.mem = torch.zeros(self.batch_size, self.out_features)

    def update_weights(
        self,
        reward: float = 0,
    ) -> None:
        with torch.inference_mode():
            # positive reward = good
            # negative reward = bad
            delta_weights = (
                self.effective_learning_rate * self.eligibility_trace * reward
            )
            self.weighted_sum.weight += delta_weights

    def forward(
        self,
        input_spikes: torch.Tensor,
    ) -> torch.Tensor:
        with torch.inference_mode():
            # input_spikes is a vector of size [batch_size, in_features]
            # but typically there will only be 1 batch

            current = self.weighted_sum(input_spikes)  # [batch_size, out_features]

            syn = self.alpha * self.syn + current  # [batch_size, out_features]
            mem = self.beta * self.mem + syn  # [batch_size, out_features]

            output_spikes = (
                mem >= self.threshold
            ).float()  # [batch_size, out_features]

            sur_dev = 1 / ((1 + (torch.pi * (mem - self.threshold)) ** 2) * torch.pi)
            # this is also of size [batch_size, out_features]

            # print(input_spikes.unsqueeze(-1).size())
            # print(sur_dev.unsqueeze(1).size())

            delta_e = input_spikes.unsqueeze(-1) * sur_dev.unsqueeze(1)
            # print(delta_e.size())
            delta_e_sum = delta_e.sum(dim=0)
            # print(delta_e_sum.size())
            self.eligibility_trace *= self.decay_value
            # print(self.eligibility_trace.size())
            self.eligibility_trace += delta_e_sum.T

            mem = mem - self.threshold * output_spikes

            self.syn, self.mem = syn, mem

            return output_spikes


layer1 = EPropLayer(in_features=28 * 28, out_features=32)
layer2 = EPropLayer(in_features=32, out_features=10)

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


def rate_encode(tensor, num_timesteps, gain):
    from snntorch import spikegen

    return spikegen.rate(data=tensor, num_steps=num_timesteps, gain=gain)


num_epochs = 1
num_timesteps = 100
gain = 0.5

correct_or_not = []
averages = []

import matplotlib.pyplot as plt

for index, (data, targets) in enumerate(train_dataloader):
    if (index + 1) % 100 == 0:
        print(f"{index+1:,}/{len(train_dataloader):,}")

        ao100 = sum(correct_or_not[-100:]) / 100
        averages.append(ao100)

        plt.figure("Live graph")
        plt.clf()
        plt.plot(averages)
        plt.ylim(0, 1)
        plt.show(block=False)
        plt.pause(0.001)

    data = data.flatten(start_dim=1)
    layer1.reset_states()
    layer2.reset_states()

    image = rate_encode(data, num_timesteps, gain)

    ans = torch.zeros(10)

    for j in range(num_timesteps):
        spk = layer1.forward(image[j])
        # print(spk.size())
        out_dist = layer2.forward(spk)
        ans += out_dist.squeeze()

    choice = ans.argmax().item()
    correct = targets.item()

    reward = 1 if choice == correct else -1

    layer1.update_weights(reward=reward)
    layer2.update_weights(reward=reward)

    thing = 0 if reward == -1 else 1
    correct_or_not.append(thing)
