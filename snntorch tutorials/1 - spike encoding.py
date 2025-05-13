# https://github.com/jeshraghian/snntorch/blob/master/examples/tutorial_1_spikegen.ipynb

import torch
import snntorch as snn

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


batch_size = 128
data_path = "data/mnist"
num_classes = 10  # 10 classes in mnist

from torchvision import datasets, transforms

# defining a transform to standardize the images
transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ]
)

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)


from snntorch import utils

subset = 10  # reduce by a factor of 10
mnist_train = utils.data_subset(mnist_train, subset=subset)
print(f"Size of mnist_train is {len(mnist_train)}")


from torch.utils.data import DataLoader

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)


# The principle of spikes is to convey a delta (change) in information
# But MNIST is a static image, so there are 2 options to pass it:
# 1. Turn each MNIST image into a static video, passing in the full image at every timestep
# 2. Split the image into many different versions of it, each different from the other but added together to be the full image

# The module snntorch.spikegen (i.e., spike generation) contains a series of functions that simplify the conversion of data into spikes.
# There are 3 options available for spike encoding in snntorch:

# 1. Rate coding: spikegen.rate
# 2. Latency coding: spikegen.latency
# 3. Delta modulation: spikegen.delta

# How do these differ?

# Rate coding uses input features to determine spiking frequency
# Latency coding uses input features to determine spike timing
# Delta modulation uses the temporal change of input features to generate spikes

# One example of how to do rate encoding is to use each pixel as the probability that at the timestep there is a spike
# pixel brightness of 0.9 -> 0.9 probability of a spike


# # ==================================================================================================================================
# num_steps = 10  # the number of timesteps the image spans

# raw_vector = torch.ones(num_steps) * 0.5

# rate_coded_vector = torch.bernoulli(raw_vector)
# print(f"Rate coded vector: {rate_coded_vector}")


# # now for the actual thing
# from snntorch import spikegen
# import matplotlib.pyplot as plt
# import snntorch.spikeplot as splt
# from IPython.display import HTML

# data = iter(train_loader)
# data_it, targets_it = next(data)

# spike_data = spikegen.rate(data_it, num_steps=num_steps)
# print(f"Spike data size: {spike_data.size()}")
# # it's [num_steps, batch_size, c, h, w] = [10, 128, 1, 28, 28]
# spike_data_sample = spike_data[:, 0, 0]
# print(f"Spike data sample size: {spike_data_sample.size()}")
# # it's [num_steps, h, w]
# fig, ax = plt.subplots()
# anim = splt.animator(spike_data_sample, fig=fig, ax=ax)
# HTML(anim.to_html5_video())
# anim.save("temp/rate_coded_spike.mp4")

# # Before, the chance to spike was exactly 1 at the bright spots
# spike_data = spikegen.rate(data_it, num_steps=num_steps, gain=0.25)
# # gain is the factor that scales the pr to fire: [0, 1] @ 0.25 gain -> [0, 0.25]
# spike_data_sample_2 = spike_data[:, 0, 0]
# fig, ax = plt.subplots()
# anim = splt.animator(spike_data_sample_2, fig=fig, ax=ax)
# HTML(anim.to_html5_video())
# anim.save("temp/rate_coded_spike_2.mp4")

# print(f"The corresponding target for spike_data_sample is: {targets_it[0]}")


# # reconstructing the original image from the spikes
# plt.figure(facecolor="w")
# plt.subplot(1,2,1)
# plt.imshow(spike_data_sample.mean(axis = 0).reshape((28, -1)).cpu(), cmap="binary")
# plt.axis("off")
# plt.title("Gain = 1")

# plt.figure(facecolor="w")
# plt.subplot(1,2,1)
# plt.imshow(spike_data_sample_2.mean(axis = 0).reshape((28, -1)).cpu(), cmap="binary")
# plt.axis("off")
# plt.title("Gain = 0.25")

# plt.show()


# # Alternatively, a raster plot could be used
# # This would transform the image into a 2d tensor with time as the first dimension

# spike_data_sample_2 = spike_data_sample_2.reshape((num_steps, -1))

# # the raster plot
# fig = plt.figure(facecolor="w", figsize=(10, 5))
# ax = fig.add_subplot(111)
# splt.raster(spike_data_sample_2, ax, s=1.5, c="black")

# plt.title("Input Layer")
# plt.xlabel("Time step")
# plt.ylabel("Neuron Number")
# plt.show()

# idx = 210  # index into 210th neuron

# fig = plt.figure(facecolor="w", figsize=(8, 1))
# ax = fig.add_subplot(111)

# splt.raster(spike_data_sample.reshape(num_steps, -1)[:, idx].unsqueeze(1), ax, s=100, c="black", marker="|")

# plt.title("Input Neuron")
# plt.xlabel("Time step")
# plt.yticks([])
# plt.show()
# # ==================================================================================================================================

# Another method instead of using rate coding is to use latency coding
# The idea here is that the input image is the added current to the neurons
# therefore higher charges will fire faster/earlier than lower charges

# # ==================================================================================================================================
# import matplotlib.pyplot as plt

# from snntorch import spikegen
# import snntorch.spikeplot as splt
# from IPython.display import HTML

# def convert_to_time(
#     data,
#     tau=5,
#     threshold=0.01,
# ):
#     spike_time = tau * torch.log(data / (data - threshold))
#     return spike_time


# # raw_input = torch.arange(0,5,0.05)
# # spike_times = convert_to_time(raw_input)

# # plt.plot(raw_input, spike_times)
# # plt.xlabel("Input value")
# # plt.ylabel("Spike time (s)")
# # plt.show()

# data = iter(train_loader)
# data_it, targets_it = next(data)

# spike_data = spikegen.latency(
#     data_it,
#     num_steps=100,
#     tau=5,
#     threshold=0.01,
#     linear=True,
#     normalize=True,
#     clip=True,
# )
# # the linear=True argument removes the exponential decay entirely
# # the normalize=True argument normalizes the firing so that it spans all the timesteps
# # the clip=True argument gets rid of the mass firing at once of the dark background that holds no practical information

# fig = plt.figure(facecolor="w", figsize=(10, 5))
# ax = fig.add_subplot(111)
# splt.raster(spike_data[:, 0].view(100, -1), ax, s=25, c="black")

# plt.title("Input Layer")
# plt.xlabel("Time step")
# plt.ylabel("Neuron Number")
# plt.show()

# spike_data_sample = spike_data[:, 0, 0]
# print(spike_data_sample.size())

# fig, ax = plt.subplots()
# anim = splt.animator(spike_data_sample, fig, ax)

# HTML(anim.to_html5_video())
# anim.save("temp/latency_coded_spike.mp4")

# # I don't really like Latency coding (not intuitive plus doesn't really tie in with how video works)
# # But it has an advantage over Raster coding in that each neuron fires at most once

# # ==================================================================================================================================

# The final type of spike coding is event driven
# It is the idea that spikes only happen when a significant enough change occurs
# The theory is that visual stuff is event/change driven
# I don't like this as it means that a completely stationary scene would have 0 inputs, but that's not the case with your vision
# Maybe it's a mix of event + rate?

# ==================================================================================================================================

import matplotlib.pyplot as plt
from snntorch import spikegen
from snntorch import spikeplot as splt

data = torch.Tensor([0, 1, 0, 2, 8, -20, 20, -5, 0, 1, 0])

# Plot the tensor
plt.plot(data)

plt.title("Some fake time-series data")
plt.xlabel("Time step")
plt.ylabel("Voltage (mV)")
plt.show()

spike_data = spikegen.delta(data, threshold=4)

# Create fig, ax
fig = plt.figure(facecolor="w", figsize=(8, 1))
ax = fig.add_subplot(111)

# Raster plot of delta converted data
splt.raster(spike_data, ax, c="black")

plt.title("Input Neuron")
plt.xlabel("Time step")
plt.yticks([])
plt.xlim(0, len(data))
plt.show()

spike_data = spikegen.delta(data, threshold=4, off_spike=True)

# Create fig, ax
fig = plt.figure(facecolor="w", figsize=(8, 1))
ax = fig.add_subplot(111)

# Raster plot of delta converted data
splt.raster(spike_data, ax, c="black")

plt.title("Input Neuron")
plt.xlabel("Time step")
plt.yticks([])
plt.xlim(0, len(data))
plt.show()

print(spike_data)

# ==================================================================================================================================