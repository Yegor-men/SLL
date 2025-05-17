import torch
import snntorch as snn

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


class EPropLIFLayer:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        alpha: float = 0.9,
        beta: float = 0.9,
        threshold: float = 1,
        e_trace_decay: float = 0.999,
        batch_size: int = 1,
        lr: float = 1e-2,
        device: str = "cuda",
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.e_trace_decay = e_trace_decay
        self.batch_size = batch_size
        self.lr = (1 - e_trace_decay) * lr

        self.w = torch.zeros(in_features, out_features, device=device)

        self.initialize_w()
        self.reset_states()

    def initialize_w(self):
        mean = 0
        std = (2 / self.in_features) ** 0.5
        torch.nn.init.normal_(self.w, mean=mean, std=std)

    def reset_states(self):
        self.e = torch.zeros_like(self.w)
        self.e_vec = torch.zeros_like(self.w)

        self.syn = torch.zeros(self.batch_size, self.out_features).to(self.w)
        self.mem = torch.zeros_like(self.syn)

    def update_weights(self, reward: float = 0):
        self.w += self.e * self.lr * reward

    def forward(self, prev_layer_spikes):
        assert prev_layer_spikes.size() == (
            self.batch_size,
            self.in_features,
        ), f"Previous layer spikes size mismatch, got {[s for s in prev_layer_spikes.size()]}, should be {[self.batch_size, self.in_features]}"

        prev_layer_spikes = prev_layer_spikes.to(self.w)  # [batch_size, in_features]

        # 1. weighted sum
        weighted_sum = prev_layer_spikes @ self.w
        # weighted_sum: [batch_size, out_features]

        # 2. update synaptic and membrane
        syn = self.syn * self.alpha + weighted_sum
        mem = self.mem * self.beta + syn
        spikes = (mem >= self.threshold).float()
        # syn, mem, spikes: [batch_size, out_features]

        # ─── Eligibility‐vector update ───
        # self.e_vec holds ε^{t-1} ∈ ℝ^{N_in×N_out}

        # 3. get prev_layer_spikes vector z_prev that matches e_vec’s shape, averaging across the batch
        z_mean = prev_layer_spikes.mean(dim=0)
        # z_mean: [in_features]
        z_prev = z_mean.unsqueeze(-1)
        # z_prev: [in_features, 1]

        # broadcast z_prev → [in_features, out_features] when adding to e_vec
        # 4. fast eligibility vector ε^t = α·ε^{t-1} + z_prev
        eps_prev = self.e_vec
        # eps_prev: [in_features, out_features]
        self.e_vec = (self.alpha * eps_prev + z_prev).detach()
        # self.e_vec (ε^t): [in_features, out_features]

        # ─── Surrogate derivative ψ_j^t ───
        # need one psi per output neuron j
        # use either triangle or exponential surrogate derivative:
        delta = 0.1
        # compute per‐neuron voltage (average across the batch)
        v_j = mem.mean(dim=0)
        # v_j: [out_features]
        # psi = torch.clamp(1 - torch.abs((v_j - self.threshold) / delta), min=0.0)
        psi = torch.exp(-(((v_j - self.threshold) / delta) ** 2))
        # psi: [out_features]

        # ─── Accumulate into slow trace ē^t ───
        # e_t = ψ_j^t * ε^t_{ji} for each (i,j)
        e_t = self.e_vec * psi.unsqueeze(0)
        # e_t: [in_features, out_features]

        # decay & add:  ē^t = γ·ē^{t-1} + e_t
        self.e = (self.e * self.e_trace_decay + e_t).detach()
        # self.e (ē^t): [in_features, out_features]

        # 5. spike, override the states and return
        mem -= spikes * self.threshold
        self.syn, self.mem = syn, mem

        return spikes


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


layer1 = EPropLIFLayer(
    in_features=28 * 28,
    out_features=10,
    alpha=0.999,
    beta=0.999,
    threshold=1,
    e_trace_decay=0.999,
    batch_size=1,
    lr=1e-3,
    device="cuda",
)

import matplotlib.pyplot as plt

num_epochs = 10
num_timesteps = 100
gain = 0.1

correct_or_not = []
averages = []

before = layer1.w.clone()

for epoch in range(num_epochs):
    with torch.inference_mode():
        for index, (data, targets) in enumerate(train_dataloader):
            if data.size(0) != batch_size:
                continue

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

            data = data.flatten(start_dim=1)
            layer1.reset_states()

            image = rate_encode(data, num_timesteps, gain)

            ans = torch.zeros(batch_size, layer1.out_features).to("cuda")

            for j in range(num_timesteps):
                spk = layer1.forward(image[j])
                ans += spk.squeeze()

            choices = ans.argmax(dim=1)
            correct = targets.to("cuda")

            num_matches = (choices == correct).sum().item()

            probability = ans[0][correct] / 100

            reward = ((num_matches * 2 - 1) * probability).item()
            # print(reward.item())
            # + if correct - if wrong

            layer1.update_weights(reward=reward)

            correct_or_not.append(num_matches)

after = layer1.w.clone()

print((after - before).sum())
plt.plot(averages)
plt.ylim(0, 1)
plt.show()
