import torch as T

from vae import modeling


def main() -> None:
    config = modeling.BaseVAEConfig.from_path("config/vae/cnn.yml")
    vae = modeling.VAE.from_config(config)
    print(vae)
    x = T.randn(2, 3, 64, 64)
    print(vae(x))


if __name__ == "__main__":
    main()
