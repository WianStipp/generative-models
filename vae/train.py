import torch as T
from torch.utils import data as torch_data
from datetime import datetime
import random
from torch.utils import tensorboard
from torchvision.utils import make_grid


from vae import modeling, data

N_EPOCHS = 3
BATCH_SIZE = 64


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = tensorboard.SummaryWriter("runs/vae-trainer-{}".format(timestamp))

    device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")
    config = modeling.BaseVAEConfig.from_path("config/vae/cnn.yml")
    vae = modeling.VAE.from_config(config).to(device).train()
    train = data.get_celeba_split("train")
    validation = data.get_celeba_split("valid")
    test = data.get_celeba_split("test")
    train_loader = torch_data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = torch_data.DataLoader(validation, batch_size=BATCH_SIZE)
    optimizer = T.optim.AdamW(vae.parameters(), lr=3e-4)

    def train_epoch_step(epoch_idx: int) -> None:
        vae.train()
        running_loss = 0.0
        for i, (batch, _) in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device)
            loss, _ = vae(batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 10 == 9:
                last_loss = running_loss / 10  # loss per batch
                print("  batch {} loss: {}".format(i + 1, last_loss))
                tb_x = epoch_idx * len(train_loader) + i + 1
                writer.add_scalar("Loss/train", last_loss, tb_x)
                running_loss = 0.0

    @T.no_grad()
    def validation_step(epoch_idx: int) -> None:
        vae.eval()
        running_loss = 0.0
        for i, (batch, _) in enumerate(validation_loader):
            batch = batch.to(device)
            loss, _ = vae(batch)
            running_loss += loss.item()
        avg_loss = running_loss / (i + 1)
        writer.add_scalar("Loss/validation", avg_loss, epoch_idx)

    @T.inference_mode()
    def generate_step(epoch_idx: int) -> None:
        vae.eval()
        test_cases = [x for x, y in random.choices(test, k=5)]
        original_grid = make_grid(test_cases, nrow=5, normalize=True, scale_each=True)
        writer.add_image("Original Images", original_grid, global_step=epoch_idx)
        test_cases_tensor = T.stack(test_cases).to(device)
        _, reconstructed = vae(test_cases_tensor)
        reconstructed_grid = make_grid(
            reconstructed.cpu(), nrow=5, normalize=True, scale_each=True
        )
        writer.add_image(
            "Reconstructed Images", reconstructed_grid, global_step=epoch_idx
        )

    for epoch in range(N_EPOCHS):
        validation_step(epoch)
        train_epoch_step(epoch)
        generate_step(epoch)


if __name__ == "__main__":
    main()
