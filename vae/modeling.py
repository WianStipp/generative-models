from typing import Tuple, List, NamedTuple
import abc
import pydantic
import torch as T
from torch import nn


class BaseVAEConfig(pydantic.BaseModel, abc.ABC):
    latent_dims: int
    activation_fn: str = "ReLU"
    dropout_rate: float = 0.1


class CNNVAEConfig(BaseVAEConfig):
    input_shape: Tuple[int, ...]
    channels: List[int]
    kernel_sizes: List[int]
    strides: List[int]


class EncoderOutput(NamedTuple):
    mu: T.Tensor
    log_Var: T.Tensor


class BaseEncoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, x: T.Tensor) -> EncoderOutput: ...


class BaseDecoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, z: T.Tensor) -> T.Tensor: ...


class VAE(nn.Module):
    def __init__(self, encoder: BaseEncoder, decoder: BaseDecoder) -> None:
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: T.Tensor) -> T.Tensor: ...
