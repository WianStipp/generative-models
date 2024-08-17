"""Some functions to visualize VAE reconstructions, latent spaces, etc."""

import torch as T
import torch.nn as nn
from torchvision.transforms import ToTensor
from PIL import Image

from ae import vae_modeling, ae_training

device = T.device("cuda:0")
encoder = vae_modeling.VAEFaceEncoder()
decoder = vae_modeling.VAEFaceDecoder()
model = vae_modeling.FaceVAE(encoder, decoder)
model = nn.DataParallel(model)
model.to(device)
model.load_state_dict(T.load(f'saved_models/vae_20230520_epoch33.pt'))
model.eval()
image = ToTensor()(Image.open("wian.jpg"))
image = image.to(device)
for i in range(1000):
  z, _, _ = model.module.encoder(image.unsqueeze(0))
  ae_training.save_original_reconstructed_images_celeba(model, [(image, "")], "wian_test", f"epoch33_gen{i}")
