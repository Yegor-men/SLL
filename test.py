import torch

foo = torch.normal(2, 3, size=(1, 4))
# print(foo)

bar = torch.zeros(5,5)
torch.nn.init.normal_(bar, mean=0, std=1)
print(bar)

print(bar.t())
print(bar.T)