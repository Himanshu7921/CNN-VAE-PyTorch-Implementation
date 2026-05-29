from matplotlib import pyplot as plt
import numpy as np
import torch
from model import CNNVAE
from config import config
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_data(dataset_path: str = config["dataset_path"]):
    """
    Load the training dataset and create a data loader.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # RGB → 1 channel
        transforms.Resize((config["img_size"], config["img_size"])),                 # Resize to 64x64
        transforms.ToTensor()                        # Convert to [0, 1] tensor
    ])

    dataset = datasets.ImageFolder(
        root = dataset_path,
        transform=transform
    )

    loader = DataLoader(dataset, batch_size = config["batch_size"], shuffle=True)
    return loader


def save_model(save_path, model, optimizer):
    """
    Load the training dataset and create a data loader.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "latent_dim": config["latent_dim"],
            "input_channels": config["input_channels"],
            "input_size": config["img_size"],
            "base_filters": config["base_filters"],
            "architecture": "4-layer-CNN"
        }
    }
    torch.save(checkpoint, save_path)
    print(f"Model and config saved successfully to {save_path}")

def load_model(device, saved_model_path = config["saved_models_path"]):
    """
    Load a trained VAE model from a checkpoint.
    """
    load_path = saved_model_path
    checkpoint = torch.load(load_path, map_location = device)

    saved_config = checkpoint["config"]
    latent_dim = saved_config["latent_dim"]

    loaded_model = CNNVAE(latent_dim=latent_dim).to(device)
    loaded_model.load_state_dict(checkpoint["model_state_dict"])

    loaded_model.eval()

    print("Model architecture re-initialized and weights loaded successfully.")
    print(f"Config used: {saved_config}")
    return loaded_model

def generate_new_images(model, n_images = 16, image_size = config["img_size"], device = "cpu"):
    """
    Generate and visualize images by sampling from the latent space.
    """
    model.to(device)
    model.eval()

    latent_dim = config["latent_dim"]

    cols = int(np.sqrt(n_images))
    rows = int(np.ceil(n_images / cols))

    _, ax = plt.subplots(rows, cols, figsize=(10, 10))

    with torch.no_grad():

        for i in range(n_images):

            z = torch.randn(1, latent_dim, device=device)

            logits = model.decoder(z)

            img = logits[0,0].cpu()

            r = i // cols
            c = i % cols

            ax[r,c].imshow(img, cmap="gray")
            ax[r,c].axis("off")
            
    plt.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=1,
        wspace=0,
        hspace=0
    )

    plt.show()