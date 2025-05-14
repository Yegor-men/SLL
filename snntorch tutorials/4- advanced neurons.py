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
            alpha=alpha, beta=beta, threshold=threshold, reset_mechanism=reset_mechanism
        )

        self.syn_1, self.mem_1 = self.lif_1.reset_mem()
        self.syn_2, self.mem_2 = self.lif_2.reset_mem()
        self.syn_3, self.mem_3 = self.lif_3.reset_mem()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        spikes = []

        num_timesteps = x.size(0)
        for timestep in range(num_timesteps):
            spk1, self.syn_1, self.mem_1 = self.lif_1(x[timestep], self.syn_1, self.mem_1)
            foo = self.ws_1(spk1)
            spk2, self.syn_2, self.mem_2 = self.lif_2(foo, self.syn_2, self.mem_2)
            bar = self.ws_2(spk2)
            spk3, self.syn_3, self.mem_3 = self.lif_3(bar, self.syn_3, self.mem_3)
            spikes.append(spk3)
        
        spikes = torch.stack(spikes)
        return spikes

def rate_encode(tensor):
    from snntorch import spikegen
    return spikegen.rate_conv(tensor)

snn = SNN(
    num_inputs=10,
    num_hidden=100,
    num_outputs=10,
    alpha=0.9,
    beta=0.9,
)

test_input = torch.rand(100, 1, 10)

spiked_input = rate_encode(test_input)

spikes = snn(spiked_input)

print(spikes.size())
print(spikes.sum(dim=0))