# import torch

# lr = 1e-3

# alphas = [0.1, 0.5, 0.9, 0.99]

# for alpha in alphas:
#     foo = torch.ones(1)
#     temp = torch.zeros_like(foo)

#     for i in range(100000):
#         temp = temp * alpha + foo

#     effective_lr = lr * (1 - alpha)

#     print(f"Alpha of {alpha}: {temp * effective_lr}")

import torch

foo = torch.tensor([[1, 2]])
bar = torch.tensor([[1, 2, 3], [4, 5, 6]])

grr = foo @ bar
print(f"{foo.size()} @ {bar.size()} = {grr.size()}")
print(grr)