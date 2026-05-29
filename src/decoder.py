import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNDecoder(nn.Module):
    """
    Convolutional decoder network for reconstructing images from latent
    representations in a Variational Autoencoder (VAE).

    Architecture
    ------------
    The decoder maps a latent vector z ∈ R^{latent_dim} into an image
    through a learned projection followed by a sequence of transposed
    convolutional upsampling blocks.

    Latent Space:
        z → Linear(latent_dim, 512 × 4 × 4)

    Reshape:
        (B, 8192) → (B, 512, 4, 4)

    Upsampling Path:
        4 × 4   →  8 × 8   (512 → 256)
        8 × 8   → 16 × 16  (256 → 128)
        16 × 16 → 32 × 32  (128 → 64)
        32 × 32 → 64 × 64  (64 → 1)

    Each intermediate upsampling stage consists of:
        ConvTranspose2d → BatchNorm2d → ReLU

    The final layer applies a Sigmoid activation to constrain pixel
    intensities to the range [0, 1], making the decoder suitable for
    grayscale image reconstruction tasks.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent representation sampled from the
        approximate posterior q(z|x).

    Input
    -----
    z : torch.Tensor
        Latent tensor of shape (batch_size, latent_dim).

    Output
    ------
    x_hat : torch.Tensor
        Reconstructed image tensor of shape (batch_size, 1, 64, 64)
        with values in the range [0, 1].

    Notes
    -----
    - Designed for 64×64 single-channel image generation.
    - Mirrors the encoder architecture by progressively increasing
      spatial resolution while decreasing channel depth.
    - The decoder parameterizes the likelihood model p(x|z) within
      the Variational Autoencoder framework.
    """
    def __init__(self, latent_dim):
        super().__init__()
        
        # Start from 4x4 bottleneck
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        
        # Layer 1: 4 -> 8
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        
        # Layer 2: 8 -> 16
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Layer 3: 16 -> 32
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Layer 4: 32 -> 64
        self.deconv4 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        """
        Decode latent vectors into reconstructed images.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation of shape (batch_size, latent_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed image tensor of shape
            (batch_size, 1, 64, 64).

        Notes
        -----
        The latent vector is first projected into a learned 4×4 feature
        map and then progressively upsampled using transposed convolution
        blocks. A Sigmoid activation is applied at the output layer to
        produce valid pixel intensities in the range [0, 1].
        """
        h = self.fc(z)
        h = h.view(-1, 512, 4, 4)
        
        h = F.relu(self.bn1(self.deconv1(h)))
        h = F.relu(self.bn2(self.deconv2(h)))
        h = F.relu(self.bn3(self.deconv3(h)))
        
        return torch.sigmoid(self.deconv4(h))