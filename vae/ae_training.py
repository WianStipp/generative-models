"""Messy training script for a vanilla autoencoder"""

import tqdm
import torch as T
import torch.optim as optim
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from vae import ae_modeling
from PIL import Image
import os
import numpy as np

batch_size = 256
learning_rate = 0.001
num_epochs = 25


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


# Load Fashion MNIST dataset
train_dataset = FashionMNIST(root="./data",
                             train=True,
                             download=True,
                             transform=ToTensor())
test_dataset = FashionMNIST(root="./data",
                            train=False,
                            download=True,
                            transform=ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = T.device("cuda:0")
# Initialize the autoencoder model
# model = ae_modeling.AutoEncoder()
from vae import vae_modeling

encoder = vae_modeling.VAEEncoder()
decoder = ae_modeling.AEDecoder()
model = vae_modeling.VAE(encoder, decoder)
model.to(device)

# Define loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in tqdm.tqdm(range(num_epochs), total=num_epochs):
  running_loss = 0.0
  for images, _ in tqdm.tqdm(train_loader,
                             total=len(train_dataset) // batch_size):
    images = images.to(device)
    # Forward pass
    prediction, loss = model(images)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

  # Print epoch statistics
  print(
      f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}"
  )

print("Training finished!")

# Save the model
T.save(model.state_dict(), 'saved_models/vae.pt')

# Set the model to evaluation mode
model.eval()

# Specify the folder to save the reconstructed images
save_folder = "./reconstructed_images"

# Make predictions and save the reconstructed images
save_original_reconstructed_images(model, test_dataset, save_folder)

print("Reconstructed images saved!")
