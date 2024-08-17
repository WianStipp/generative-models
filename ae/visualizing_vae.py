"""Some functions to visualize VAE reconstructions, latent spaces, etc."""

import matplotlib.pyplot as plt
import torch as T
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from ae import vae_modeling, ae_modeling

device = T.device("cuda:0")
encoder = vae_modeling.VAEEncoder()
decoder = ae_modeling.AEDecoder()
model = vae_modeling.VAE(encoder, decoder)
model.to(device)
model.load_state_dict(T.load('saved_models/vae.pt'))
model.eval()

test_dataset = FashionMNIST(root="./data",
                            train=False,
                            download=True,
                            transform=ToTensor())

latent_points = []
labels = []
for i, (image, label) in enumerate(test_dataset):
  image = image.to(device)
  z, _, _ = model.encoder(image.unsqueeze(0))
  latent_points.append(tuple(*z.detach().cpu().numpy()))
  labels.append(label)
fig = plt.figure(figsize=(12, 8))
x, y = zip(*latent_points)
plt.scatter(x=x, y=y, c=labels, alpha=0.8, s=10)
fig.savefig("vae.png")
