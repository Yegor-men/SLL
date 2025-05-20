import torch
import time

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.set_grad_enabled(False)


class ALIFLayer:
    def __init__(
        self,
        num_in: int,
        num_out: int,
        e_trace_decay: float = 0.99,
        learning_rate: float = 1e-3,
        device: str = "cuda",
    ):
        self.num_in = num_in
        self.num_out = num_out
        self.e_trace_decay = e_trace_decay
        self.lr = learning_rate

        # 1. weight matrix w and it's eligibility buffers
        self.w = torch.randn(num_in, num_out).to(device)
        self.eps_syn = torch.zeros_like(self.w)  # syn decay per synapse
        self.eps_mem = torch.zeros_like(self.w)  # mem decay per synapse
        self.e_trace = torch.zeros_like(self.w)  # running total, e_trace

        # 2. raw values for alpha and beta (pre-sigmoid) and threshold
        self.raw_alpha = torch.zeros(num_out).to(device)
        self.raw_beta = torch.zeros(num_out).to(device)
        self.threshold = torch.ones(num_out).to(device)

        # 3. eligibility buffers for alpha for each neuron
        self.eta_syn = torch.zeros_like(self.raw_alpha)  # η_j^syn
        self.eta_mem = torch.zeros_like(self.raw_alpha)  # η_j^mem
        self.e_trace_alpha = torch.zeros_like(self.raw_alpha)  # ∑_t e_{α_j}^t

        # 4. eligibility buffers for beta for each neuron
        self.zeta_mem = torch.zeros_like(self.raw_beta)  # ζ_j^mem
        self.e_trace_beta = torch.zeros_like(self.raw_beta)  # ∑_t e_{β_j}^t

        # 5. eligibility buffers for the threshold for each neuron
        self.e_trace_threshold = torch.zeros_like(self.threshold)  # ∑_t e_{thresh_j}^t

        self.syn = torch.zeros(num_out).to(device)
        self.mem = torch.zeros(num_out).to(device)

    def reset_states(self):
        self.eps_syn.zero_()
        self.eps_mem.zero_()
        self.e_trace.zero_()

        self.eta_syn.zero_()
        self.eta_mem.zero_()
        self.e_trace_alpha.zero_()

        self.zeta_mem.zero_()
        self.e_trace_beta.zero_()

        self.e_trace_threshold.zero_()

        self.syn.zero_()
        self.mem.zero_()

    def update_parameters(self, reward: float = 0.0, baseline: float = 0.0):
        net_reward = reward - baseline

        # 1. weight update: Δw = η * (r - b) * ∑_t e_{ij}^t
        self.w += self.lr * net_reward * self.e_trace * (1 - self.e_trace_decay)

        # 2. update raw alpha
        # Δα_raw[j] = η (r-b) (∑_t e_{α_j}^t) * σ'(α_raw[j])
        alpha = torch.sigmoid(self.raw_alpha)
        alpha_derivative = alpha * (1 - alpha)  # sigmoid derivative
        self.raw_alpha += self.lr * net_reward * self.e_trace_alpha * alpha_derivative

        # 3. update raw beta
        beta = torch.sigmoid(self.raw_beta)
        beta_derivative = beta * (1 - beta)  # sigmoid derivative
        self.raw_beta += self.lr * net_reward * self.e_trace_beta * beta_derivative

        # 4. threshold update
        # Δthreshold[j] = η (r-b) ∑_t (- f'(mem_j^t - thresh_j))
        self.threshold += self.lr * net_reward * self.e_trace_threshold

    def forward(self, in_spikes: torch.Tensor):
        assert in_spikes.size() == (
            self.num_in,
        ), f"Expected size {[self.num_in]}, received {[dim for dim in in_spikes.size()]} instead"

        in_spikes = in_spikes.to(self.w)

        # 1. calculate alpha and beta values from the raw values
        alpha = torch.sigmoid(self.raw_alpha)  # [out]
        beta = torch.sigmoid(self.raw_beta)  # [out]

        # 2. calculate the net current
        c_net = in_spikes @ self.w  # [out]

        # 3. update syn and mem
        new_syn = alpha * self.syn + c_net  # [out]
        new_mem = beta * self.mem + new_syn  # [out]

        # 5. calculate spikes and surrogate derivative
        u = new_mem - self.threshold  # [out]
        out_spikes = (u >= 0.0).float()
        # surrogate derivative f'(u) = 1 / [π (1 + u^2)]
        gradient = 1.0 / (torch.pi * (1.0 + u * u))  # shape [out]
        gradient_broadcast = gradient.unsqueeze(0)  # shape [1, out] for broadcasting

        # 6. Weight‐eligibility updates for each synapse (i→j)
        # eps_syn[i,j] = α_j * eps_syn[i,j] + s_i^t
        # eps_mem[i,j] = β_j * eps_mem[i,j] + eps_syn[i,j]
        # e_ij^t = gradient[j] * eps_mem[i,j]
        # accumulate sum_t e_ij^t → e_trace[i,j]
        # broadcast alpha, beta, and in_spikes to shape [in, out]
        alpha_mat = alpha.unsqueeze(0)  # [1, out], broadcast to [in, out]
        beta_mat = beta.unsqueeze(0)  # [1, out]
        spikes_mat = in_spikes.unsqueeze(1)  # [in, 1], broadcast to [in, out]

        # 6a. update eps_syn and eps_mem
        self.eps_syn = alpha_mat * self.eps_syn + spikes_mat  # [in, out]
        self.eps_mem = beta_mat * self.eps_mem + self.eps_syn  # [in, out]

        # 6b. calculate scalar eligibility e_ij^t and accumulate
        e_ij_t = gradient_broadcast * self.eps_mem  # [in, out]
        self.e_trace = self.e_trace * self.e_trace_decay + e_ij_t

        # 7. alpha eligibility per neuron
        # η_syn[j] = α_j * η_syn[j] + syn_j^{t-1}
        # η_mem[j] = β_j * η_mem[j] + η_syn[j]
        # e_{α_j}^t = slope[j] * η_mem[j], accumulate into e_trace_alpha[j]
        self.eta_syn = alpha * self.eta_syn + self.syn  # [out]
        self.eta_mem = beta * self.eta_mem + self.eta_syn  # [out]
        e_alpha_t = gradient * self.eta_mem  # [out]
        self.e_trace_alpha = self.e_trace_alpha * self.e_trace_decay + e_alpha_t

        # 8. beta eligibility per neuron
        # ζ_mem[j] = mem_j^{t-1} + β_j * ζ_mem[j]
        # e_{β_j}^t = slope[j] * ζ_mem[j], accumulate into e_trace_beta[j]
        self.zeta_mem = self.mem + beta * self.zeta_mem  # [out]
        e_beta_t = gradient * self.zeta_mem  # [out]
        self.e_trace_beta = self.e_trace_beta * self.e_trace_decay + e_beta_t

        # 9. threshold eligibility per neuron
        # e_{thresh_j}^t = - slope[j], accumulate into e_trace_thresh[j]
        e_thresh_t = -gradient  # [out]
        self.e_trace_threshold = (
            self.e_trace_threshold * self.e_trace_decay + e_thresh_t
        )

        self.syn = new_syn
        self.mem = new_mem - self.threshold * out_spikes

        return out_spikes


class SNN:
    def __init__(
        self,
        e_trace_decay: float = 0.99,
        learning_rate: float = 1e-3,
        gain: float = 0.5,
    ):
        self.gain = gain

        self.layers = [
            ALIFLayer(
                num_in=28 * 28,
                num_out=32,
                e_trace_decay=e_trace_decay,
                learning_rate=learning_rate,
            ),
            ALIFLayer(
                num_in=32,
                num_out=32,
                e_trace_decay=e_trace_decay,
                learning_rate=learning_rate,
            ),
            ALIFLayer(
                num_in=32,
                num_out=10,
                e_trace_decay=e_trace_decay,
                learning_rate=learning_rate,
            ),
        ]

    def rate_code(self, tensor: torch.Tensor) -> torch.Tensor:
        from snntorch import spikegen

        return spikegen.rate(tensor, num_steps=1, gain=self.gain).squeeze()

    def update_parameters(self, reward: float) -> None:
        for i, layer in enumerate(self.layers):
            layer.update_parameters(reward=reward)

    def reset_states(self) -> None:
        for i, layer in enumerate(self.layers):
            layer.reset_states()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rate_code(x)
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)

        return x


# ====================================================================================================

snn = SNN()

test_image = torch.rand(28 * 28)
summed = snn.forward(test_image)
summed.zero_()

# ====================================================================================================

# start = time.time()

# # for i in range(1000):
# #     out = snn.forward(test_image)
# #     summed += out
# # summed /= summed.sum()

# end = time.time()

# print(summed.argmax())
# print(f"{end-start:5f}s")


# ====================================================================================================

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

batch_size = 1
num_timesteps = 100
data_path = "data/mnist"

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

num_epochs = 1

correct_or_not = []
cum_sums = []

loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    print(f"Epoch {(epoch+1):,}/{num_epochs:,} - {((epoch+1)/num_epochs)*100:.2f}%")

    for index, (image, label) in enumerate(train_dataloader):
        summed.zero_()
        image = image.flatten()
        for i in range(num_timesteps):
            out = snn.forward(image)
            summed += out
        distribution = (summed / summed.sum()).to("cpu")
        base_reward = int(label == distribution.argmax()) * 2 - 1
        loss = loss_fn(distribution.unsqueeze(0), label)
        effective_reward = base_reward * loss
        snn.update_parameters(reward=effective_reward)
        snn.reset_states()

        correct_or_not.append((base_reward + 1) / 2)

        if len(correct_or_not) % 100 == 0:
            cum_sum = sum(correct_or_not[-100:]) / 100
            cum_sums.append(cum_sum)
            plt.figure("Live graph")
            plt.clf()
            plt.plot(cum_sums)
            plt.ylim(0, 1)
            plt.show(block=False)
            plt.pause(0.1)

            print(
                f"\t{index+1:,}/{len(train_dataloader):,} batches {((index+1)/len(train_dataloader))*100:.2f}%"
            )
