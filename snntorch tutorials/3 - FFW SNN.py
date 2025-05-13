# https://github.com/jeshraghian/snntorch/blob/master/examples/tutorial_3_feedforward_snn.ipynb

import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
from torch import nn
import matplotlib.pyplot as plt

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


# @title Plotting Settings
def plot_cur_mem_spk(
    cur,
    mem,
    spk,
    thr_line=False,
    vline=False,
    title=False,
    ylim_max1=1.25,
    ylim_max2=1.25,
):
    # Generate Plots
    fig, ax = plt.subplots(
        3, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1, 0.4]}
    )

    # Plot input current
    ax[0].plot(cur, c="tab:orange")
    ax[0].set_ylim([0, ylim_max1])
    ax[0].set_xlim([0, 200])
    ax[0].set_ylabel("Input Current ($I_{in}$)")
    if title:
        ax[0].set_title(title)

    # Plot membrane potential
    ax[1].plot(mem)
    ax[1].set_ylim([0, ylim_max2])
    ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")
    if thr_line:
        ax[1].axhline(
            y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2
        )
    plt.xlabel("Time step")

    # Plot output spike using spikeplot
    splt.raster(spk, ax[2], s=400, c="black", marker="|")
    if vline:
        ax[2].axvline(
            x=vline,
            ymin=0,
            ymax=6.75,
            alpha=0.15,
            linestyle="dashed",
            c="black",
            linewidth=2,
            zorder=0,
            clip_on=False,
        )
    plt.ylabel("Output spikes")
    plt.yticks([])

    plt.show()


def plot_snn_spikes(spk_in, spk1_rec, spk2_rec, title):
    # Generate Plots
    fig, ax = plt.subplots(
        3, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1, 0.4]}
    )

    # Plot input spikes
    splt.raster(spk_in[:, 0], ax[0], s=0.03, c="black")
    ax[0].set_ylabel("Input Spikes")
    ax[0].set_title(title)

    # Plot hidden layer spikes
    splt.raster(spk1_rec.reshape(num_steps, -1), ax[1], s=0.05, c="black")
    ax[1].set_ylabel("Hidden Layer")

    # Plot output spikes
    splt.raster(spk2_rec.reshape(num_steps, -1), ax[2], c="black", marker="|")
    ax[2].set_ylabel("Output Spikes")
    ax[2].set_ylim([0, 10])

    plt.show()


# Now doing it the proper way as a layer and whatnot


def leaky_integrate_and_fire(mem, x, w, beta, threshold=1):
    spk = mem > threshold  # if membrane exceeds threshold, spk=1, else, 0
    mem = beta * mem + w * x - spk * threshold
    return spk, mem


# set neuronal parameters
delta_t = torch.tensor(1e-3)
tau = torch.tensor(5e-3)
beta = torch.exp(-delta_t / tau)

print(f"The decay rate is: {beta:.3f}")

num_steps = 200

# initialize inputs/outputs + small step current input
x = torch.cat((torch.zeros(10), torch.ones(190) * 0.5), 0)
mem = torch.zeros(1)
spk_out = torch.zeros(1)
mem_rec = []
spk_rec = []

# neuron parameters
w = 0.4
beta = 0.819

# neuron simulation
for step in range(num_steps):
    spk, mem = leaky_integrate_and_fire(mem, x[step], w=w, beta=beta)
    mem_rec.append(mem)
    spk_rec.append(spk)

# convert lists to tensors
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)

plot_cur_mem_spk(
    x * w,
    mem_rec,
    spk_rec,
    thr_line=1,
    ylim_max1=0.5,
    title="LIF Neuron Model With Weighted Step Voltage",
)


# The above code is built into snn as snn.Leaky

lif1 = snn.Leaky(beta=0.8, threshold=1)

# Which expects the exact same inputs and outputs as for Lapicque except it's easier because only 2 hyperparameters
# 1. cur_in for the incoming current
# 2. mem for the membrane potential before
#
# 1. spk_out for the outward spikes
# 2. mem for the updated membrane

w = 0.21
cur_in = torch.cat((torch.zeros(10), torch.ones(190) * w), 0)
mem = torch.zeros(1)
spk = torch.zeros(1)
mem_rec = []
spk_rec = []

# neuron simulation
for step in range(num_steps):
    spk, mem = lif1(cur_in[step], mem)
    mem_rec.append(mem)
    spk_rec.append(spk)

# convert lists to tensors
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)

plot_cur_mem_spk(
    cur_in, mem_rec, spk_rec, thr_line=1, ylim_max1=0.5, title="snn.Leaky Neuron Model"
)


# Now to make an actual neural net
# pytorch handles the weights and parameters
# snntorch handles the neurons
# Effectively pytorch will manage the synapse connection strengths and whatnot
# While snntorch decides what to pass to the next layer


class SNN(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_hidden,
        num_outputs,
        beta,
    ):
        super().__init__()

        self.fc1 = nn.Linear(in_features=num_inputs, out_features=num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(in_features=num_hidden, out_features=num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()

        self.mem2_rec = []
        self.spk1_rec = []
        self.spk2_rec = []

    def forward(self, x) -> None:
        num_timesteps = x.size(0)

        for timestep in range(num_timesteps):
            cur1 = self.fc1(x[timestep])
            spk1, self.mem1 = self.lif1(cur1, self.mem1)
            cur2 = self.fc2(spk1)
            spk2, self.mem2 = self.lif2(cur2, self.mem2)

            self.mem2_rec.append(self.mem2)
            self.spk1_rec.append(spk1)
            self.spk2_rec.append(spk2)

        self.mem2_rec = torch.stack(self.mem2_rec)
        self.spk1_rec = torch.stack(self.spk1_rec)
        self.spk2_rec = torch.stack(self.spk2_rec)

        plot_snn_spikes(
            x, self.spk1_rec, self.spk2_rec, "Fully Connected Spiking Neural Network"
        )


num_inputs = 784
num_hidden = 1000
num_outputs = 10
beta = 0.99

snn = SNN(
    num_inputs=num_inputs,
    num_hidden=num_hidden,
    num_outputs=num_outputs,
    beta=beta,
)

# snntorch ALWAYS uses time-first dimensionality: [time, batch, everything else]
spk_in = spikegen.rate_conv(torch.rand((200, 784))).unsqueeze(1)
# unsqueeze at dim=1 to simulate a batch size
print(f"Dimension of spk_in: {spk_in.size()}")

snn(spk_in)

# from IPython.display import HTML

# fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
# labels=['0', '1', '2', '3', '4', '5', '6', '7', '8','9']
# spk2_rec = snn.spk2_rec.squeeze(1).detach().cpu()

# # plt.rcParams['animation.ffmpeg_path'] = 'C:\\path\\to\\your\\ffmpeg.exe'

# #  Plot spike count histogram
# anim = splt.spike_count(spk2_rec, fig, ax, labels=labels, animate=True)
# HTML(anim.to_html5_video())
# anim.save("temp/spike_bar.mp4")

