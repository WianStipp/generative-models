"""Some generic classes, functions used in autoencoder variants"""

from typing import NamedTuple
import torch as T


class VAEOutput(NamedTuple):
  output: T.Tensor  # e.g. reconstruction
  loss: T.Tensor
