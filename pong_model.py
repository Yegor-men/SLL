import torch
from torch import nn
import snntorch as snn


class LIFLayer(nn.Module):
    def __init__(
        self,
        out_features: int,
        alpha: float = 0.9,
        beta: float = 0.9,
    ):
        super().__init__()

        self.ws = nn.LazyLinear(out_features=out_features, bias=False)
        self.lif = snn.RSynaptic(alpha=alpha, beta=beta, linear_features=out_features, reset_mechanism="zero")

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lif.reset_mem()

    def forward(self, x) -> torch.Tensor:
        cur = self.ws(x)
        spk, syn, mem = self.lif(cur, self.lif.spk, self.lif.syn, self.lif.mem)
        self.lif.spk, self.lif.syn, self.lif.mem = spk, syn, mem
        print(mem)
        return spk


class SNN(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Flatten(),
            LIFLayer(200),
            LIFLayer(100),
            LIFLayer(2),
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.layers(image)
        logits_sum = logits.sum()
        if logits_sum == 0:
            logits_sum = 1
        probabilities = logits / logits_sum
        return probabilities

# snn = SNN().to("cuda")

# for i in range(100):
#     test_image = torch.randn(1, 3, 512, 512).to("cuda")
#     out = snn(test_image)
#     print(out)