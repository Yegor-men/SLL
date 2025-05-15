import torch
import matplotlib.pyplot as plt


def scale_tensor(
    tensor: torch.Tensor,
    fps: float = 50,
    min_freq: float = 1,
    max_freq: float = 10,
) -> torch.Tensor:
    """
    Scales a tensor so that given an FPS, a minimal and maximal firing frequency,
    The output tensor is clamped so that when converted to a spike input it will be noise + the actual image
    """
    min_clamp = min_freq / fps
    max_clamp = max_freq / fps
    return tensor * (max_clamp - min_clamp) + min_clamp


def convert_to_spikes(
    tensor: torch.Tensor,
):
    from snntorch import spikegen

    return spikegen.rate_conv(data=tensor)


def render_image(tensor: torch.Tensor) -> None:
    assert tensor.size(0) == 3, "there aren't 3 color channels"

    # Move channels to last, CPU & NumPy
    img = tensor.permute(1, 2, 0).cpu().numpy()

    plt.figure("Live Image")  # gives/uses a window named "Live Image"
    plt.clf()  # clear the current figure
    plt.imshow(img)
    plt.axis("off")
    plt.show(block=False)  # non‐blocking
    plt.pause(0.001)  # let GUI update


temp = torch.zeros(3, 512, 512)
for i in range(100):
    test = torch.rand(3, 512, 512)
    test = scale_tensor(test)
    test = convert_to_spikes(test)
    render_image(test)
