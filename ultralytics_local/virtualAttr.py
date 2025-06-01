import io
import matplotlib.pyplot as plot
import numpy as np
import cv2
import einops
from PIL import Image
import torch
from torchvision.models._utils import IntermediateLayerGetter
from torchvision import transforms

from ultralytics import YOLO

model = YOLO("last.pt")

model.eval()
sub_model = IntermediateLayerGetter(model.model.model, {"1": "vit1"})
print(sub_model)

p = Image.open("../ultralytics/tests/datasets/DOTAv1.5/images/train/images/P0000.jpg")

transform = transforms.Compose([
    transforms.ToTensor(),
])

transform1 = transforms.Compose([
    transforms.CenterCrop((1024, 1024)),
    transforms.Normalize(mean=torch.tensor((0.0, 0.0, 0.0)), std=torch.tensor((1.0, 1.0, 1.0))),
])

resize = transforms.Resize((256, 256))

p = transform(p)
p = p[:, 0:1024, 0:1024]
raw = einops.rearrange(resize(p), "c h w -> h w c")
raw = raw[:, :, [2, 1, 0]]
p = transform1(p)
print(p.shape)

blue = torch.zeros(1, 1, 3)
blue = blue
red = torch.zeros(1, 1, 3)
red = red
red[0, 0, 0] = 1

out = sub_model(p.reshape(1, 3, 1024, 1024))['vit1']
out = einops.reduce(out, "b c h w -> h w", reduction="mean")
out = einops.reduce(out, "h w -> h w 3", reduction="repeat")
out = torch.lerp(blue, red, out)
print(out.shape)

out = raw + out

cv2.imshow("", out.numpy())
cv2.waitKey(0)
cv2.destroyAllWindows()
