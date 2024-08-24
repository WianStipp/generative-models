from typing import NamedTuple
import torch as T
from torchvision import datasets, transforms
import PIL.Image

DATA_FOLDER = "data/"


class CelebAVAETransform:
    """Transformation of CelebA data for VAE training"""

    def __init__(self, dims: int = 64) -> None:
        self.transform = transforms.Compose(
            [
                transforms.Resize(dims, antialias=True),
                transforms.CenterCrop(dims),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, image: PIL.Image) -> T.Tensor:
        return self.transform(image)


class CelebADatasets(NamedTuple):
    train: datasets.CelebA
    validation: datasets.CelebA
    test: datasets.CelebA


def get_celeba_datasets(root_folder: str = DATA_FOLDER) -> CelebADatasets:
    train = get_celeba_split("train", root_folder)
    validation = get_celeba_split("valid", root_folder)
    test = get_celeba_split("test", root_folder)
    return CelebADatasets(train, validation, test)


def get_celeba_split(split: str, root_folder: str = DATA_FOLDER) -> datasets.CelebA:
    return datasets.CelebA(
        root=root_folder, split=split, transform=CelebAVAETransform(), download=True
    )


def main() -> None:
    train, validation, test = get_celeba_datasets()
    print(f"Training dataset size: {len(train)}")
    print(f"Validation dataset size: {len(validation)}")
    print(f"Testing dataset size: {len(test)}")


if __name__ == "__main__":
    main()
