import torch
from tqdm import tqdm
from model import CNNVAE
from config import config
from utils import load_data, save_model
import torch.nn.functional as F

def train_vae_model(model, loader, epochs, optimizer, device, anneal_epochs = config["beta_anneal_epochs"]):
    """
    Train a Variational Autoencoder (VAE) using reconstruction and KL
    divergence losses with beta annealing.

    Parameters
    ----------
    model : CNNVAE
        Variational Autoencoder model.
    loader : DataLoader
        Training data loader.
    epochs : int
        Number of training epochs.
    optimizer : torch.optim.Optimizer
        Optimization algorithm.
    device : str
        Training device ('cpu' or 'cuda').
    anneal_epochs : int
        Number of epochs used for KL annealing.

    Returns
    -------
    CNNVAE
        Trained VAE model.
    """
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_recon_loss = 0
        total_kl_loss = 0
        
        # Beta annealing for KL Divergence
        beta = min(1.0, epoch / anneal_epochs)
        
        # tqdm setup with a description
        batch_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for images, _ in batch_bar:
            x = images.to(device)

            # Forward pass
            (mean, log_var), reconstructed_x = model(x)
            
            # 1. Reconstruction Loss (Binary Cross Entropy)
            recon_loss = F.binary_cross_entropy(reconstructed_x, x, reduction="mean")
            
            # 2. KL Divergence Loss
            kl_val = model.kl_divergence(mean, log_var)
            kl_loss = torch.mean(kl_val)

            combined_loss = recon_loss + (kl_loss * beta * 0.0001)

            # Backward pass
            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()

            # Stats for logging
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

            # Update tqdm bar with both losses
            batch_bar.set_postfix({
                "Recon": f"{recon_loss.item():.4f}",
                "KL": f"{kl_loss.item():.4f}",
                "Beta": f"{beta:.2f}"
            })

        avg_recon = total_recon_loss / len(loader)
        avg_kl = total_kl_loss / len(loader)
        
        print(f"==> Epoch {epoch} Complete | Avg Recon: {avg_recon:.4f} | Avg KL: {avg_kl:.4f}")

    return model

def main():
    """
    Train the VAE and save the learned model parameters.

    Workflow
    --------
    1. Load dataset.
    2. Initialize model and optimizer.
    3. Train the VAE.
    4. Save model checkpoint.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = load_data(dataset_path = config["dataset_path"])

    # Get the CNN VAE model and Optimizer
    model = CNNVAE(latent_dim = config["latent_dim"])
    optimizer = torch.optim.Adam(params = model.parameters(), lr = config["lr"])

    # Pass model, Optimizer and  necessary stuffs to train the vae model
    print(f"Training on device: [{device}]")
    trained_vae = train_vae_model(
        model = model,
        loader = loader,
        epochs = config["epochs"],
        optimizer = optimizer,
        device = device
    )

    # save the model
    save_model(save_path = config["saved_models_path"], model = trained_vae, optimizer = optimizer)

if __name__ == "__main__":
    main()