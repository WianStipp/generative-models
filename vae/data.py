from typing import NamedTuple
import torch as T
from torchvision import datasets, transforms
import PIL.Image

DATA_FOLDER = "data/"


class CelebAVAETransform:
    """Transformation of CelebA data for VAE training"""

    def __init__(self, dims: int = 128) -> None:
        self.piltotensor = transforms.ToTensor()
        self.resizer = transforms.Resize((dims, dims))

    def __call__(self, image: PIL.Image) -> T.Tensor:
        return self.resizer(self.piltotensor(image))


class CelebADatasets(NamedTuple):
    train: datasets.CelebA
    validation: datasets.CelebA
    test: datasets.CelebA


def get_celeba_datasets(root_folder: str = DATA_FOLDER) -> CelebADatasets:
    train = datasets.CelebA(
        root=root_folder, split="train", transform=CelebAVAETransform(), download=True
    )
    validation = datasets.CelebA(
        root=root_folder, split="valid", transform=CelebAVAETransform(), download=True
    )
    test = datasets.CelebA(
        root=root_folder, split="test", transform=CelebAVAETransform(), download=True
    )
    return CelebADatasets(train, validation, test)


def main() -> None:
    train, validation, test = get_celeba_datasets()
    print(f"Training dataset size: {len(train)}")
    print(f"Validation dataset size: {len(validation)}")
    print(f"Testing dataset size: {len(test)}")


if __name__ == "__main__":
    main()
