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
        decay_value: float = 0.999,
        learning_rate: float = 1e-3,
        batch_size: int = 1,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.beta = beta
        self.decay_value = decay_value
        self.effective_learning_rate = learning_rate * (1 - decay_value)
        self.batch_size = 1

        self.weighted_sum = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
        )
        # nn.Linear has .weight, the weight matrix of size [in_features, out_features]

        self.eligibility_trace = torch.zeros_like(self.weighted_sum.weight)
        # eligibility trace is the same size as weights to be able to update them, of size [in_features, out_features]

        self.threshold = 1.0

        self.syn = torch.zeros(batch_size, out_features)
        self.mem = torch.zeros(batch_size, out_features)

    def update_weights(
        self,
        reward: float = 0,
    ) -> None:
        # positive reward = good
        # negative reward = bad
        delta_weights = self.effective_learning_rate * self.eligibility_trace * reward
        self.weighted_sum.weight += delta_weights

    def forward(
        self,
        input_spikes: torch.Tensor,
    ) -> torch.Tensor:
        # input_spikes is a vector of size [batch_size, in_features]
        # but typically there will only be 1 batch

        current = self.weighted_sum(input_spikes)  # [batch_size, out_features]

        syn = self.alpha * self.syn + current  # [batch_size, out_features]
        mem = self.beta * self.mem + syn  # [batch_size, out_features]

        output_spikes = (mem >= self.threshold).float()  # [batch_size, out_features]

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


layer = EPropLayer(in_features=5, out_features=3)

from snntorch import spikegen

rand = torch.rand(1, 5)

test = spikegen.rate(data=rand, num_steps=100, gain=0.75)

for i in range(test.size(0)):
    out = layer.forward(test[i])

print(layer.eligibility_trace)
