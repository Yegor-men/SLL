# import torch
# from torch import nn
# import snntorch as snn

# torch.manual_seed(0)
# torch.cuda.manual_seed(0)


# class EPropLayer:
#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#         alpha: float = 0.9,
#         beta: float = 0.9,
#         threshold: float = 1,
#         eligibility_decay: float = 0.999,
#         learning_rate: float = 0,
#         batch_size: int = 1,
#     ) -> None:
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.beta = beta
#         self.threshold = threshold
#         self.decay_value = eligibility_decay
#         self.effective_learning_rate = learning_rate * (1 - eligibility_decay)
#         self.batch_size = batch_size

#         self.weighted_sum = nn.Linear(
#             in_features=in_features,
#             out_features=out_features,
#             bias=False,
#         )
#         # nn.Linear has .weight, the weight matrix of size [out_features, in_features]
#         # p.s. pytorch is goofy and decides to .T every time for some reason?? why??

#         self.reset_states()

#     def reset_states(self):
#         self.eligibility_trace = torch.zeros_like(self.weighted_sum.weight)
#         # eligibility trace is the same as weight: [out_features, in_features]

#         self.syn = torch.zeros(self.batch_size, self.out_features)
#         self.mem = torch.zeros(self.batch_size, self.out_features)

#     def update_weights(
#         self,
#         reward: float = 0,
#     ) -> None:
#         with torch.inference_mode():
#             # positive reward = good
#             # negative reward = bad
#             delta_weights = (
#                 self.effective_learning_rate * self.eligibility_trace * reward
#             )
#             self.weighted_sum.weight += delta_weights

#     def forward(
#         self,
#         input_spikes: torch.Tensor,
#     ) -> torch.Tensor:
#         with torch.inference_mode():
#             # input_spikes is a vector of size [batch_size, in_features]
#             # but typically there will only be 1 batch

#             current = self.weighted_sum(input_spikes)  # [batch_size, out_features]

#             syn = self.alpha * self.syn + current  # [batch_size, out_features]
#             mem = self.beta * self.mem + syn  # [batch_size, out_features]

#             output_spikes = (
#                 mem >= self.threshold
#             ).float()  # [batch_size, out_features]

#             sur_dev = 1 / ((1 + (torch.pi * (mem - self.threshold)) ** 2) * torch.pi)
#             # this is also of size [batch_size, out_features]

#             # print(input_spikes.unsqueeze(-1).size())
#             # print(sur_dev.unsqueeze(1).size())

#             delta_e = input_spikes.unsqueeze(-1) * sur_dev.unsqueeze(1)
#             # print(delta_e.size())
#             delta_e_sum = delta_e.sum(dim=0)
#             # print(delta_e_sum.size())
#             self.eligibility_trace *= self.decay_value
#             # print(self.eligibility_trace.size())
#             self.eligibility_trace += delta_e_sum.T

#             mem = mem - self.threshold * output_spikes

#             self.syn, self.mem = syn, mem

#             return output_spikes


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
        psi = torch.exp(-((v_j - self.threshold) / delta) ** 2)
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


layer = EPropLIFLayer(
    in_features=10,
    out_features=5,
    alpha=0.9,
    beta=0.9,
    threshold=1,
    e_trace_decay=0.99,
    batch_size=1,
    lr=1e-2,
    device="cuda",
)


def rate_encode(tensor, num_steps: int, gain: float = 1):
    from snntorch import spikegen

    return spikegen.rate(tensor, num_steps, gain)


test_image = torch.ones(layer.batch_size, layer.in_features)
spike_image = rate_encode(tensor=test_image, num_steps=1000, gain=0.25)

dist = torch.zeros(layer.batch_size, layer.out_features)

for i in range(spike_image.size(0)):
    out = layer.forward(spike_image[i])
    dist += out.to("cpu")

print(dist / (i + 1))

print(layer.e)

print(layer.w)
