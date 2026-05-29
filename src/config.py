"""
Training and model configuration for the CNN-VAE.

Contains hyperparameters, dataset paths, model architecture settings,
and checkpoint locations used during training and sampling.
"""

config = {
    "lr": 2e-4,
    "dataset_path": "./data/celeba",
    "latent_dim": 64,
    "img_size": 64,
    "batch_size": 64,
    "beta_anneal_epochs": 40,
    "epochs": 200,
    "saved_models_path": "./saved_model/CNN_VAE.pth",
    "base_filters": 64,
    "input_channels": 1
}