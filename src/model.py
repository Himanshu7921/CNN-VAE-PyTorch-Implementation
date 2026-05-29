import torch
import torch.nn as nn
from encoder import CNNEncoder
from decoder import CNNDecoder

class CNNVAE(nn.Module):
    """
    Convolutional Variational Autoencoder (CNN-VAE) for learning compact
    probabilistic latent representations of image data.

    Overview
    --------
    The model consists of two primary components:

        Encoder:
            x → q(z|x)

        Decoder:
            z → p(x|z)

    The encoder maps an input image to the parameters of a latent Gaussian
    distribution, while the decoder reconstructs the image from a sampled
    latent vector.

    Variational Framework
    ---------------------
    For an input image x, the encoder predicts:

        μ, logσ² = Encoder(x)

    A latent vector is then sampled using the reparameterization trick:

        z = μ + σ ⊙ ε
        ε ~ N(0, I)

    The decoder reconstructs the input image:

        x̂ = Decoder(z)

    The model is optimized using the Evidence Lower Bound (ELBO):

        ELBO = E_q(z|x)[log p(x|z)]
            - D_KL(q(z|x) || p(z))

    where the prior distribution is assumed to be:

        p(z) = N(0, I)

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space.

    Input
    -----
    x : torch.Tensor
        Input image tensor of shape
        (batch_size, channels, height, width).

    Outputs
    -------
    (mean, log_var) : Tuple[torch.Tensor, torch.Tensor]
        Parameters of the approximate posterior distribution q(z|x).

    logits : torch.Tensor
        Reconstructed image produced by the decoder.

    Notes
    -----
    - Implements an end-to-end variational autoencoder architecture.
    - Uses convolutional feature extraction for image encoding.
    - Employs the reparameterization trick for differentiable latent
    sampling.
    - Assumes a standard Gaussian latent prior.
    - Suitable for representation learning, image reconstruction,
    generative modeling, and latent space exploration.
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.encoder = CNNEncoder(latent_dim)
        self.decoder = CNNDecoder(latent_dim)
    
    def forward(self, x: torch.tensor):
        """
        Perform a complete forward pass through the Variational Autoencoder.

        The input image is encoded into the parameters of a latent Gaussian
        distribution. A latent sample is then drawn using the reparameterization
        trick and passed through the decoder to generate a reconstruction.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor.

        Returns
        -------
        Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]

            (mean, log_var)
                Parameters of the approximate posterior distribution q(z|x).

            logits
                Reconstructed image generated from the sampled latent vector.

        Notes
        -----
        The returned latent statistics are required for computing the KL
        divergence component of the ELBO objective during training.
        """
        mean, log_var = self.encoder(x)
        z = self.encoder.reparameterization(mean, log_var)
        logits = self.decoder(z)
        return (mean, log_var), logits
    
    def kl_divergence(self, mu: torch.tensor, log_var: torch.tensor):
        """
        Compute the Kullback-Leibler divergence between the learned latent
        distribution q(z|x) and the standard Gaussian prior p(z).

        The closed-form solution for two diagonal Gaussian distributions is

            D_KL(q(z|x) || p(z))
            =
            -1/2 Σ [1 + logσ² - σ² - μ²]

        where

            q(z|x) = N(μ, diag(σ²))
            p(z)   = N(0, I)

        Parameters
        ----------
        mu : torch.Tensor
            Mean vector of the approximate posterior distribution.

        log_var : torch.Tensor
            Log-variance vector of the approximate posterior distribution.

        Returns
        -------
        torch.Tensor
            Per-sample KL divergence values with shape (batch_size,).

        Notes
        -----
        - Encourages the learned latent distribution to remain close to the
        standard Gaussian prior.
        - Acts as a regularization term in the ELBO objective.
        - Larger KL values indicate greater deviation from the prior
        distribution.
        """
        return -0.5 * torch.sum(1 + log_var - torch.exp(log_var) - mu.pow(2), dim=-1)