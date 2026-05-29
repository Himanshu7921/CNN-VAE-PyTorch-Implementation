from config import config
from utils import load_model, generate_new_images
import torch


def sample():
    """
    Load a trained VAE and generate synthetic images.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device = device, saved_model_path = config["saved_models_path"])
    generate_new_images(model, n_images = 100, device="cpu")

if __name__ == "__main__":
    """
    Entry point for image generation.
    """
    sample()