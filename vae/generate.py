import click
import torch as T
import numpy as np
import matplotlib.pyplot as plt

from vae import modeling


@click.command()
@click.argument("model_path")
def main(model_path: str) -> None:
    config = modeling.BaseVAEConfig.from_path("config/vae/cnn.yml")
    vae = modeling.VAE.from_config(config).eval()
    vae.load_state_dict(T.load(model_path, weights_only=True))
    grid_width, grid_height = (5, 3)
    z_sample = T.randn(size=(grid_height * grid_width, config.latent_dims))
    with T.inference_mode():
        reconstructions = (
            vae.generate_from_z(z_sample).detach().permute(0, 2, 3, 1).numpy()
        )
    generate_images(grid_width, grid_height, reconstructions)


def generate_images(
    grid_width: int, grid_height: int, reconstructions: np.ndarray
) -> T.Tensor:
    fig = plt.figure(figsize=(18, 5))
    fig.subplots_adjust(hspace=0.15, wspace=0.10)
    for i in range(grid_width * grid_height):
        ax = fig.add_subplot(grid_height, grid_width, i + 1)
        ax.axis("off")
        ax.imshow(reconstructions[i, :, :, :])
    fp = "reconstructions.png"
    print(f"saving to {fp}")
    fig.savefig(fp)


if __name__ == "__main__":
    main()
