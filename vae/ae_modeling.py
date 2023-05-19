"""This module contains an Autoencoder model"""

import torch as T
from torch import nn
import torch.nn.functional as F


class AEEncoder(nn.Module):
    """Encoder for the autoencoder."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 28, (3, 3), stride=2, padding="valid")
        self.conv2 = nn.Conv2d(28, 64, (3, 3), stride=2, padding="valid")
        self.conv3 = nn.Conv2d(64, 128, (3, 3), stride=2, padding="valid")
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(512, 2)

    def forward(self, x: T.Tensor) -> T.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        return self.dense(x)


class AEDecoder(nn.Module):
    """Decoder for the autoencoder."""

    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(2, 2048)
        self.conv1t = nn.ConvTranspose2d(128, 64, (2, 2), stride=2, padding=(0, 0))
        self.conv2t = nn.ConvTranspose2d(64, 28, (3, 3), stride=2, padding=(1, 1))
        self.conv3t = nn.ConvTranspose2d(28, 28, (2, 2), stride=2, padding=(0, 0))
        self.conv4 = nn.Conv2d(28, 1, (3, 3), 1, padding="valid")

    def forward(self, z: T.Tensor) -> T.Tensor:
        dense_out: T.Tensor = self.dense(z)
        dense_out = dense_out.reshape(-1, 128, 4, 4)
        x = F.relu(self.conv1t(dense_out))
        x = F.relu(self.conv2t(x))
        x = F.relu(self.conv3t(x))
        x = self.conv4(x)
        return x


class AutoEncoder(nn.Module):
    """Autoencoder model."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = AEEncoder()
        self.decoder = AEDecoder()

    def forward(self, image: T.Tensor) -> T.Tensor:
        return self.decoder(self.encoder(image))


if __name__ == "__main__":
    ae = AutoEncoder()
    x = T.normal(T.zeros(28, 28), T.ones(28))[None, None, :, :]
    z = ae(x)
    print(z.shape)
