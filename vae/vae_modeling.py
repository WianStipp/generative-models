"""This module contains a simple variational autoencoder.

We are assuming no covariance, so just need to estimate z_mean and z_log_var.
We can sample from the latent space using z = z_mean + exp(z_log_var * 0.5) * epsilon
(reparameterization trick.)
"""

from typing import Tuple
import torch as T
import torch.nn.functional as F
import torch.nn as nn

from vae import ae_modeling, vae_generic


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
    batch_size, dim = z_mu.shape
    z = z_mu + T.exp(z_log_var * 0.5) * T.normal(T.zeros(
      (batch_size, dim)), T.ones((batch_size, dim))).to(z_mu.device)
    return z, z_mu, z_log_var


def sample_z(z_mu: T.Tensor, z_log_var: T.Tensor, device: T.device) -> T.Tensor:
  """Sample a latent vector z from the mean and log var vectors
  using the reparameterization trick for backprop."""
  batch_size, dim = z_mu.shape
  return z_mu + T.exp(z_log_var * 0.5) * T.normal(T.zeros(
      (batch_size, dim)), T.ones((batch_size, dim))).to(device)


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
    kl_penalty = T.mean(-0.5 * T.sum(1 + z_log_var - T.square(z_mu) -
                              T.exp(z_log_var), dim=1))
    loss = reconstruction_loss + kl_penalty
    return vae_generic.VAEOutput(reconstruction, loss)


if __name__ == "__main__":
  encoder = VAEEncoder()
  decoder = ae_modeling.AEDecoder()
  vae = VAE(encoder, decoder)
  x = T.normal(T.zeros(28, 28), T.ones(28))[None, None, :, :]
  x_pred, loss = vae(x)
  print(f"{loss=}")
  print(x_pred.shape)
