"""Messy training script for a vanilla autoencoder"""

from typing import Tuple
from collections import deque
import wandb
import tqdm
import click
import torch as T
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import FashionMNIST, CelebA
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from ae import ae_modeling
from PIL import Image
import os
import numpy as np

from ae import vae_modeling, vae_generic

learning_rate = 0.0005


def save_original_reconstructed_images(model, test_dataset, save_folder):
  if not os.path.exists(save_folder):
    os.makedirs(save_folder)

  device = next(
      model.parameters()).device  # Get the device of the model parameters

  for i, (image, label) in enumerate(test_dataset):
    # Move image tensor to the same device as the model
    if i >= 10:
      break
    image = image.to(device)

    # Forward pass
    output, _ = model(image.unsqueeze(0))
    reconstructed_image = output.squeeze(0).detach().cpu()

    # Convert tensors to NumPy ndarrays
    original_image = image.squeeze(0).cpu().numpy()
    reconstructed_image = reconstructed_image.numpy()

    # Reshape the tensors to (28, 28)
    original_image = np.clip(original_image, 0, 1)
    reconstructed_image = np.squeeze(reconstructed_image)
    reconstructed_image = np.clip(reconstructed_image, 0, 1)

    # Scale the pixel values to [0, 255]
    original_image = (original_image * 255).astype(np.uint8)
    reconstructed_image = (reconstructed_image * 255).astype(np.uint8)

    # Create PIL Images from the NumPy arrays
    original_image = Image.fromarray(original_image)
    reconstructed_image = Image.fromarray(reconstructed_image)

    # Save the original image with label in the filename
    original_filename = f"original_{label}_{i}.png"
    original_save_path = os.path.join(save_folder, original_filename)
    original_image.save(original_save_path)

    # Save the reconstructed image with label in the filename
    reconstructed_filename = f"reconstructed_{label}_{i}.png"
    reconstructed_save_path = os.path.join(save_folder, reconstructed_filename)
    reconstructed_image.save(reconstructed_save_path)

def save_original_reconstructed_images_celeba(model, test_dataset, save_folder, filename_prefix: str = ''):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    device = next(model.parameters()).device  # Get the device of the model parameters

    for i, (image, _) in enumerate(test_dataset):
        # Move image tensor to the same device as the model
        if i > 100:
          break
        image = image.to(device)

        # Forward pass
        output, _ = model(image.unsqueeze(0))
        reconstructed_image = output.squeeze(0).detach().cpu()

        # Convert tensors to NumPy ndarrays
        original_image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        reconstructed_image = reconstructed_image.permute(1, 2, 0).numpy()

        # Scale the pixel values to [0, 255]
        original_image = (original_image * 255).astype(np.uint8)
        reconstructed_image = (reconstructed_image * 255).astype(np.uint8)

        # Create PIL Images from the NumPy arrays
        original_image = Image.fromarray(original_image)
        reconstructed_image = Image.fromarray(reconstructed_image)

        # Save the original image with a sequential filename
        original_filename = f"original_{i:06d}.png"
        original_save_path = os.path.join(save_folder, original_filename)
        original_image.save(original_save_path)

        # Save the reconstructed image with a sequential filename
        reconstructed_filename = f"{filename_prefix}reconstructed_{i:06d}.png"
        reconstructed_save_path = os.path.join(save_folder, reconstructed_filename)
        reconstructed_image.save(reconstructed_save_path)

def select_datasets(dataset_name: str) -> Tuple[Dataset, Dataset]:
  if dataset_name == 'fashion_mnist':
    train_dataset = FashionMNIST(root="./data",
                                 train=True,
                                 download=True,
                                 transform=ToTensor())
    test_dataset = FashionMNIST(root="./data",
                                train=False,
                                download=True,
                                transform=ToTensor())
  elif dataset_name == 'celeba':
    # load and rehape CelebA images to 64 * 64
    train_dataset = CelebA(root="/mnt/data/torch_datasets",
                           split='train',
                           download=True,
                           transform=vae_generic.ToResizedTensor((64, 64)))
    test_dataset = CelebA(root="/mnt/data/torch_datasets",
                          split='test',
                          download=True,
                          transform=vae_generic.ToResizedTensor((64, 64)))
  else:
    raise ValueError(f"{dataset_name=} not supported.")
  return (train_dataset, test_dataset)


def select_model(model_name: str) -> nn.Module:
  if model_name == 'fashion_mnist_autoencoder':
    return ae_modeling.AutoEncoder()
  elif model_name == 'fashion_mnist_vae':
    encoder = vae_modeling.VAEEncoder()
    decoder = ae_modeling.AEDecoder()
    return vae_modeling.VAE(encoder, decoder)
  elif model_name == 'celeba_vae':
    encoder = vae_modeling.VAEFaceEncoder()
    decoder = vae_modeling.VAEFaceDecoder()
    return vae_modeling.FaceVAE(encoder, decoder)
  raise ValueError(f"{model_name=} not supported.")


@click.command()
@click.option('--dataset_name')
@click.option('--model_name')
@click.option('--num_epochs', default=2, type=int)
@click.option('--batch_size', default=256, type=int)
def main(dataset_name: str, model_name: str, num_epochs: int,
         batch_size: int) -> None:
  wandb.init(project='face_vae')
  train_dataset, test_dataset = select_datasets(dataset_name)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  device = T.device("cuda")
  model = select_model(model_name)
  model = model.to(device)
  model = nn.DataParallel(model)
  model.to(device)

  # Define loss function and optimizer
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  # Training loop
  running_losses = deque(maxlen=50)
  for epoch in tqdm.tqdm(range(num_epochs), total=num_epochs):
    running_loss = 0.0
    for images, _ in tqdm.tqdm(train_loader,
                               total=len(train_dataset) // batch_size):
      images = images.to(device)
      # Forward pass
      _, loss = model(images)
      # Backward and optimize
      optimizer.zero_grad()
      loss = T.mean(loss)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      running_losses.append(loss.item())
      avg_loss = sum(running_losses) / len(running_losses)
      wandb.log({"avg_loss": avg_loss})

    # Print epoch statistics
    wandb.log({"epoch": epoch+1, "loss": running_loss / len(train_loader)})
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}"
    )
    T.save(model.state_dict(), f'saved_models/vae_20230520_epoch{epoch+1}.pt')
  print("Training finished!")
  # Save the model
  wandb.log({"Training Finished": True})
  T.save(model.state_dict(), 'saved_models/vae_20230520.pt')
  # Set the model to evaluation mode
  # model.load_state_dict(T.load('saved_models/vae.pt'))
  model.eval()
  # Specify the folder to save the reconstructed images
  save_folder = "./reconstructed_images"
  # Make predictions and save the reconstructed images
  save_original_reconstructed_images_celeba(model, test_dataset, save_folder)
  print("Reconstructed images saved!")


if __name__ == "__main__":
  main()
