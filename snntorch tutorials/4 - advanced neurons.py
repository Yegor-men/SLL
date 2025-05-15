# https://github.com/jeshraghian/snntorch/blob/master/examples/tutorial_4_advanced_neurons.ipynb

# seems like I'll be using snn.Leaky and snn.Synaptic most the time, synaptic seems more advanced/real

import torch
from torch import nn
import snntorch as snn

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


class SNN(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_hidden: int,
        num_outputs: int,
        alpha: float = 0.9,
        beta: float = 0.9,
        threshold: float = 1,
        reset_mechanism: str = "subtract",
    ) -> None:
        super().__init__()

        self.num_outputs = num_outputs

        self.lif_1 = snn.Synaptic(
            alpha=alpha, beta=beta, threshold=threshold, reset_mechanism=reset_mechanism
        )
        self.ws_1 = nn.Linear(
            in_features=num_inputs, out_features=num_hidden, bias=False
        )
        self.lif_2 = snn.Synaptic(
            alpha=alpha, beta=beta, threshold=threshold, reset_mechanism=reset_mechanism
        )
        self.ws_2 = nn.Linear(
            in_features=num_hidden, out_features=num_outputs, bias=False
        )
        self.lif_3 = snn.Synaptic(
            alpha=alpha,
            beta=beta,
            threshold=threshold,
            reset_mechanism=reset_mechanism,
            # inhibition=True,
        )

        self.syn_1, self.mem_1 = self.lif_1.reset_mem()
        self.syn_2, self.mem_2 = self.lif_2.reset_mem()
        self.syn_3, self.mem_3 = self.lif_3.reset_mem()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        spikes = torch.zeros(x.size(1), self.num_outputs)
        # batch size and num outputs
        print(spikes.size())

        num_timesteps = x.size(0)
        for timestep in range(num_timesteps):
            spk1, self.syn_1, self.mem_1 = self.lif_1(
                x[timestep], self.syn_1, self.mem_1
            )
            foo = self.ws_1(spk1)
            spk2, self.syn_2, self.mem_2 = self.lif_2(foo, self.syn_2, self.mem_2)
            bar = self.ws_2(spk2)
            spk3, self.syn_3, self.mem_3 = self.lif_3(bar, self.syn_3, self.mem_3)
            spikes += spk3

        row_sums = spikes.sum(dim=1, keepdim=True)
        safe_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        probabilities = spikes / safe_sums

        return probabilities


def rate_encode(tensor):
    from snntorch import spikegen

    return spikegen.rate_conv(tensor)


num_timesteps = 100
batch_size = 16

num_inputs = 10000
num_hidden = 10000
num_outputs = 5

snn = SNN(
    num_inputs=num_inputs,
    num_hidden=num_hidden,
    num_outputs=num_outputs,
    alpha=0.9,
    beta=0.9,
)

test_input = torch.rand(num_timesteps, batch_size, num_inputs)

spiked_input = rate_encode(test_input)

import time

start = time.time()
spikes = snn(spiked_input)
end = time.time()

print(spikes)
print(f"Operation took: {end-start:.5f}s")
