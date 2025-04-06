import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import thop

from ultralytics.formerBackbone.blocks.vitblk import ViTBlock5

dataset = MNIST(root='./data', download=True, transform=transforms.Compose([transforms.ToTensor()]))
print(dataset)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)

m1 = nn.Sequential(
    nn.Conv2d(1, 4, 3, 1),
    ViTBlock5(4, 8, 1),
    ViTBlock5(8, 16, 2),
    ViTBlock5(16, 32, 2),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(32, 10)
)

m2 = nn.Sequential(
    nn.Conv2d(1, 4, 3),
    nn.Conv2d(4, 8, 3),
    nn.Conv2d(8, 16, 3, 2),
    nn.Conv2d(16, 32, 3, 2),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(32, 10)
)


print(thop.profile(m1, inputs=(torch.randn(1, 1, 28, 28),), ret_layer_info=True))
print(thop.profile(m2, inputs=(torch.randn(1, 1, 28, 28),), ret_layer_info=True))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(m1.parameters(), lr=0.001)

for epoch in range(5):
    loss = None
    for data in dataloader:
        inputs, labels = data
        outputs = m1(inputs)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(loss.item())
torch.save(m1,'m1.pt')

optimizer2 = torch.optim.AdamW(m2.parameters(), lr=0.001)

for epoch in range(5):
    loss = None
    for data in dataloader:
        inputs, labels = data
        outputs = m2(inputs)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer2.step()
        optimizer2.zero_grad()
    print(loss.item())
torch.save(m2,'m2.pt')
