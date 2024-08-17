"""This module contains a simple variational autoencoder.

We are assuming no covariance, so just need to estimate z_mean and z_log_var.
We can sample from the latent space using z = z_mean + exp(z_log_var * 0.5) * epsilon
(reparameterization trick.)
"""

from typing import Tuple
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from ae import ae_modeling, vae_generic


class VAEEncoder(nn.Module):
  """Encoder for a variational autoencoder"""

  def __init__(self) -> None:
    super().__init__()
    self.conv1 = nn.Conv2d(1, 28, (3, 3), stride=2, padding='valid')
    self.conv2 = nn.Conv2d(28, 56, (3, 3), stride=2, padding='valid')
    self.conv3 = nn.Conv2d(56, 102, (3, 3), stride=2, padding='valid')
    self.flatten = nn.Flatten()
    self.z_mu_dense = nn.Linear(408, 2)
    self.z_log_var_dense = nn.Linear(408, 2)

  def forward(self, image: T.Tensor) -> Tuple[T.Tensor, T.Tensor, T.Tensor]:
    x = self.conv1(image)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    z_mu = self.z_mu_dense(x)
    z_log_var = self.z_log_var_dense(x)
    z = vae_generic.sample_z(z_mu, z_log_var)
    return z, z_mu, z_log_var


class VAE(nn.Module):
  """A simple variational autoencoder."""

  def __init__(self, encoder: VAEEncoder,
               decoder: ae_modeling.AEDecoder) -> None:
    super().__init__()
    self.beta = 500
    self.encoder = encoder
    self.decoder = decoder
    self.loss = nn.MSELoss()

  def forward(self, image: T.Tensor) -> vae_generic.VAEOutput:
    z, z_mu, z_log_var = self.encoder(image)
    reconstruction = self.decoder(z)
    reconstruction_loss = self.beta * T.sqrt(self.loss(image, reconstruction))
    kl_penalty = T.mean(
        -0.5 * T.sum(1 + z_log_var - T.square(z_mu) - T.exp(z_log_var), dim=1))
    loss = reconstruction_loss + kl_penalty
    return vae_generic.VAEOutput(reconstruction, loss)


class VAEFaceEncoder(nn.Module):
  """A variational autoencoder to encode human faces."""

  def __init__(self) -> None:
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=3,
                           out_channels=128,
                           kernel_size=3,
                           stride=2,
                           padding='valid')
    self.batchnorm1 = nn.BatchNorm2d(128)
    self.conv2 = nn.Conv2d(in_channels=128,
                           out_channels=128,
                           kernel_size=3,
                           stride=2,
                           padding='valid')
    self.batchnorm2 = nn.BatchNorm2d(128)
    self.conv3 = nn.Conv2d(in_channels=128,
                           out_channels=128,
                           kernel_size=3,
                           stride=2,
                           padding='valid')
    self.batchnorm3 = nn.BatchNorm2d(128)
    self.conv4 = nn.Conv2d(in_channels=128,
                           out_channels=128,
                           kernel_size=3,
                           stride=2,
                           padding='valid')
    self.batchnorm4 = nn.BatchNorm2d(128)
    self.flatten = nn.Flatten()
    self.z_mu_dense = nn.Linear(1152, 200)
    self.z_log_var_dense = nn.Linear(1152, 200)

  def forward(self, image: T.Tensor) -> T.Tensor:
    x = self.conv1(image)
    x = self.batchnorm1(x)
    x = F.leaky_relu(x)
    x = self.conv2(x)
    x = self.batchnorm2(x)
    x = F.leaky_relu(x)
    x = self.conv3(x)
    x = self.batchnorm3(x)
    x = F.leaky_relu(x)
    x = self.conv4(x)
    x = self.batchnorm4(x)
    x = F.leaky_relu(x)
    x = self.flatten(x)
    z_mu = self.z_mu_dense(x)
    z_log_var = self.z_log_var_dense(x)
    z = vae_generic.sample_z(z_mu, z_log_var)
    return z, z_mu, z_log_var


class VAEFaceDecoder(nn.Module):
  """A variational autoencoder to decode human faces."""

  def __init__(self) -> None:
    super().__init__()
    self.dense1 = nn.Linear(200, 1152)
    # self.batchnorm1 = nn.BatchNorm1d(1152)
    self.convt1 = nn.ConvTranspose2d(in_channels=128,
                                     out_channels=128,
                                     kernel_size=3,
                                     padding=(1, 1),
                                     stride=2)
    self.batchnorm2 = nn.BatchNorm2d(128)
    self.convt2 = nn.ConvTranspose2d(
        in_channels=128,
        out_channels=128,
        kernel_size=3,
        stride=2,
        padding=(1, 1),
    )
    self.batchnorm3 = nn.BatchNorm2d(128)
    self.convt3 = nn.ConvTranspose2d(in_channels=128,
                                     out_channels=128,
                                     kernel_size=3,
                                     stride=2,
                                     padding=(1, 1))
    self.batchnorm4 = nn.BatchNorm2d(128)
    self.convt4 = nn.ConvTranspose2d(in_channels=128,
                                     out_channels=128,
                                     kernel_size=2,
                                     padding=(1, 1),
                                     stride=2)
    self.batchnorm5 = nn.BatchNorm2d(128)
    self.convt5 = nn.ConvTranspose2d(in_channels=128,
                                     out_channels=3,
                                     kernel_size=2,
                                     stride=2)

  def forward(self, z: T.Tensor) -> T.Tensor:
    x = self.dense1(z)
    x = F.leaky_relu(x)
    x = x.reshape(-1, 128, 3, 3)
    x = self.convt1(x)
    x = self.batchnorm2(x)
    x = F.leaky_relu(x)
    x = self.convt2(x)
    x = self.batchnorm3(x)
    x = F.leaky_relu(x)
    x = self.convt3(x)
    x = self.batchnorm4(x)
    x = F.leaky_relu(x)
    x = self.convt4(x)
    x = self.batchnorm5(x)
    x = F.leaky_relu(x)
    x = self.convt5(x)
    return x


class FaceVAE(nn.Module):
  """A variational autoencoder to generate human faces."""

  def __init__(self,
               encoder: VAEFaceEncoder,
               decoder: VAEFaceDecoder,
               beta: float = 2000.0) -> None:
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.beta = beta
    self.loss = nn.MSELoss()

  def forward(self, image: T.Tensor) -> vae_generic.VAEOutput:
    z, z_mu, z_log_var = self.encoder(image)
    reconstruction = self.decoder(z)
    reconstruction_loss = self.beta * self.loss(image, reconstruction)
    kl_penalty = T.mean(
        -0.5 * T.sum(1 + z_log_var - T.square(z_mu) - T.exp(z_log_var), dim=1))
    loss = reconstruction_loss + kl_penalty
    return vae_generic.VAEOutput(reconstruction, loss)


if __name__ == "__main__":
  encoder = VAEFaceEncoder()
  decoder = VAEFaceDecoder()
  vae = FaceVAE(encoder, decoder)
  d = 64
  # need to add channels
  r = T.normal(T.zeros(d, d), T.ones(d))[None, None, :, :]
  g = T.normal(T.zeros(d, d), T.ones(d))[None, None, :, :]
  b = T.normal(T.zeros(d, d), T.ones(d))[None, None, :, :]
  rgb = T.concat((r, g, b), dim=1)
  x, loss = vae(rgb)
  print(x.shape)
  print(loss)
