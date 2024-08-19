import torch as T
from torch.utils import data as torch_data
from datetime import datetime
from torch.utils import tensorboard


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
    train_loader = torch_data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = T.optim.AdamW(vae.parameters(), lr=3e-4)

    def train_epoch_step(epoch_idx: int) -> None:
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

    for epoch in range(N_EPOCHS):
        train_epoch_step(epoch)


if __name__ == "__main__":
    main()
