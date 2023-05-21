"""Some generic classes, functions used in autoencoder variants"""

from typing import NamedTuple, Sequence
import torch as T
import torchvision.transforms.functional as F
from torchvision.transforms import Resize, ToTensor


class VAEOutput(NamedTuple):
  output: T.Tensor  # e.g. reconstruction
  loss: T.Tensor


def sample_z(z_mu: T.Tensor, z_log_var: T.Tensor) -> T.Tensor:
  """Sample a latent vector z from the mean and log var vectors
  using the reparameterization trick for backprop."""
  batch_size, dim = z_mu.shape
  return z_mu + T.exp(z_log_var * 0.5) * T.normal(T.zeros(
      (batch_size, dim)), T.ones((batch_size, dim))).to(z_mu.device)


class ToResizedTensor:

  def __init__(self, size: int | Sequence) -> None:
    """
    Args:
      size: sequence or int
    """
    self.size = size

  def __call__(self, pic):
    """
    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to reshaped tensor.

    Returns:
        Tensor: Converted image.
    """

    pic_as_tensor = ToTensor()(pic)
    return Resize(size=self.size)(pic_as_tensor)
