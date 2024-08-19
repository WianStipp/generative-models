from typing import Tuple, List, NamedTuple, Literal, Union, cast
import pydantic_yaml
import abc
import pydantic
import torch as T
from torch import nn
import torch.nn.functional as F


class BaseVAEConfig(pydantic.BaseModel, abc.ABC):
    latent_dims: int
    activation_fn: str = "ReLU"
    dropout_rate: float = pydantic.Field(default=0.1, ge=0.0, le=1.0)

    @classmethod
    def from_path(cls, path: str) -> "BaseVAEConfig":

        class _ConfigParser(pydantic.BaseModel):
            config: Union[tuple(cls.__subclasses__())] = pydantic.Field(  # type: ignore
                ..., discriminator="type_"
            )

        return pydantic_yaml.parse_yaml_file_as(_ConfigParser, path).config


class CNNVAEConfig(BaseVAEConfig):
    type_: Literal["CNN"] = "CNN"
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


class CNNEncoder(BaseEncoder):
    def __init__(self, config: CNNVAEConfig) -> None:
        super().__init__()
        self.config = config
        layers = self._create_layers()
        self.layers = nn.Sequential(*layers)
        self._set_output_dims()
        self.mu_projection = nn.Linear(self.output_dims, config.latent_dims)
        self.logvar_projection = nn.Linear(self.output_dims, config.latent_dims)

    def _create_layers(self) -> List[nn.Module]:
        config = self.config
        in_channels, *_ = config.input_shape
        layers: List[nn.Module] = []
        for out_channels, kernel_size, stride in zip(
            config.channels, config.kernel_sizes, config.strides
        ):
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            )
            layers.append(getattr(nn, self.config.activation_fn)())
            layers.append(nn.Dropout2d(self.config.dropout_rate))
            in_channels = out_channels
        return layers

    @T.no_grad()
    def _set_output_dims(self) -> None:
        dummy_input = T.zeros(1, *self.config.input_shape)
        dummy_output = self.layers(dummy_input)
        self.output_dims = dummy_output.numel() // dummy_output.size(0)

    def forward(self, x: T.Tensor) -> EncoderOutput:
        x = self.layers(x)
        x = x.flatten(start_dim=1)
        return EncoderOutput(self.mu_projection(x), self.logvar_projection(x))


class BaseDecoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, z: T.Tensor) -> T.Tensor: ...


class CNNDecoder(BaseDecoder):
    def __init__(self, config: CNNVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(
            self.config.latent_dims, self.config.channels[-1]
        )
        layers = self._create_layers()
        self.layers = nn.Sequential(*layers)

    def _create_layers(self) -> List[nn.Module]:
        config = self.config
        layers: List[nn.Module] = []
        in_channels = config.channels[-1]
        for out_channels, kernel_size, stride in zip(
            config.channels[-2::-1], config.kernel_sizes[-2::-1], config.strides[-2::-1]
        ):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            )
            layers.append(getattr(nn, self.config.activation_fn)())
            layers.append(nn.Dropout2d(self.config.dropout_rate))
            in_channels = out_channels
        layers.append(
            nn.ConvTranspose2d(
                in_channels=config.channels[0],
                out_channels=config.input_shape[0],
                kernel_size=config.kernel_sizes[0],
                stride=config.strides[0],
                output_padding=config.strides[0] - 1,
            )
        )

        layers.append(nn.Sigmoid())
        return layers

    def forward(self, z: T.Tensor) -> T.Tensor:
        x = self.input_projection(z)
        x = x.view(-1, self.config.channels[-1], 1, 1)
        return self.layers(x)


class VAE(nn.Module):
    def __init__(self, encoder: BaseEncoder, decoder: BaseDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        mu, log_var = cast(EncoderOutput, self.encoder(x))
        z = self.sample_z_from_mean_logvar(mu, log_var)
        x_hat = self.decoder(z)
        kl_term = T.mean(
            0.5 * T.sum(1 + log_var - T.square(mu) - T.exp(log_var), dim=1)
        )
        print(x.shape, x_hat.shape)
        loss = F.mse_loss(x_hat, x, reduction="mean") - kl_term
        return loss, x_hat

    def sample_z(self, x: T.Tensor) -> T.Tensor:
        mu, log_var = cast(EncoderOutput, self.encoder(x))
        return self.sample_z_from_mean_logvar(mu, log_var)

    @staticmethod
    def sample_z_from_mean_logvar(mu: T.Tensor, log_var: T.Tensor) -> T.Tensor:
        eps = T.randn(size=mu.shape).to(mu.device)
        z = mu + (T.sqrt(T.exp(log_var)) * eps)
        return z

    @classmethod
    def from_config(cls, config: BaseVAEConfig) -> "VAE":
        if isinstance(config, CNNVAEConfig):
            encoder = CNNEncoder(config)
            decoder = CNNDecoder(config)
            return cls(encoder, decoder)
        raise ValueError(f"Did not recognize config type: {type(config)}")
