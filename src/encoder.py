import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    """
    Convolutional encoder network for learning latent representations in a
    Variational Autoencoder (VAE).

    Architecture
    ------------
    The encoder compresses a 64×64 single-channel input image into the
    parameters of a latent Gaussian distribution q(z|x).

    Feature Extraction Path:
        64 × 64 → 32 × 32   (1 → 64)
        32 × 32 → 16 × 16   (64 → 128)
        16 × 16 → 8 × 8     (128 → 256)
        8 × 8   → 4 × 4     (256 → 512)

    Each convolutional stage consists of:
        Conv2d → BatchNorm2d → ReLU

    The final feature map is flattened and projected into two independent
    linear layers that estimate the parameters of the approximate posterior:

        μ      = f_mean(x)
        logσ²  = f_logvar(x)

    where:
        q(z|x) = N(μ, diag(σ²))

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent representation.

    Input
    -----
    x : torch.Tensor
        Input image tensor of shape (batch_size, 1, 64, 64).

    Outputs
    -------
    mu : torch.Tensor
        Mean vector of the approximate posterior distribution with shape
        (batch_size, latent_dim).

    log_var : torch.Tensor
        Log-variance vector of the approximate posterior distribution with
        shape (batch_size, latent_dim).

    Notes
    -----
    - Designed for grayscale image encoding.
    - Produces the distribution parameters required by the variational
    inference framework.
    - The learned latent representation captures a compressed probabilistic
    description of the input image.
    - The encoder parameterizes q(z|x) in the Evidence Lower Bound (ELBO)
    optimization objective.
    """
    def __init__(self, latent_dim):
        super().__init__()
        
        # Layer 1: 64 -> 32
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Layer 2: 32 -> 16
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Layer 3: 16 -> 8
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Layer 4: 8 -> 4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.flatten = nn.Flatten()
        
        # Flattened size is 512 * 4 * 4 = 8192
        self.fc_mean = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

    def forward(self, x):
        """
        Encode an input image into the parameters of a latent Gaussian
        distribution.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch_size, 1, 64, 64).

        Returns
        -------
        mu : torch.Tensor
            Mean vector of the approximate posterior distribution.

        log_var : torch.Tensor
            Log-variance vector of the approximate posterior distribution.

        Notes
        -----
        The input image is progressively compressed through a hierarchy of
        convolutional feature extraction layers. The resulting feature vector
        is transformed into the mean and log-variance parameters used to define
        the latent distribution q(z|x).
        """
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h))) # Added forward for layer 4

        h = self.flatten(h)
        mu = self.fc_mean(h)
        log_var = self.fc_logvar(h)
        return mu, log_var
    
    def reparameterization(self, mu, log_var):
        """
        Sample latent vectors using the reparameterization trick.

        Given the parameters of the approximate posterior distribution

            q(z|x) = N(μ, diag(σ²))

        a differentiable sample is generated as

            z = μ + σ ⊙ ε

        where

            ε ~ N(0, I)

        This formulation enables gradients to propagate through the sampling
        operation during backpropagation.

        Parameters
        ----------
        mu : torch.Tensor
            Mean vector of the latent distribution.

        log_var : torch.Tensor
            Log-variance vector of the latent distribution.

        Returns
        -------
        z : torch.Tensor
            Sampled latent vector of shape (batch_size, latent_dim).

        Notes
        -----
        The reparameterization trick separates stochastic sampling from the
        learnable parameters, making variational inference compatible with
        gradient-based optimization.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std