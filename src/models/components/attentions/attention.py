import torch
import torch.nn as nn
import torch.nn.functional as F

# Follow Stable Diffusion: https://nn.labml.ai/diffusion/stable_diffusion/model/autoencoder.html

class AttentionBlock(nn.Module):
    """
    ## Attention block
    """

    def __init__(self,
                 channels: int,
                 n_heads: int = None,
                 n_layers: int = None,
                 d_cond:int = None):
        """
        channels: is the number of channels
        """
        super().__init__()

        # normalization
        self.norm = nn.Sequential(nn.GroupNorm(32, channels),
                                  nn.SiLU(inplace=True))

        # Query, key and value mappings
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)

        # Final convolution layer
        self.proj_out = nn.Conv2d(channels, channels, 1)

        # Attention scaling factor
        self.scale = channels**-0.5

    def forward(self, x: torch.Tensor, cond=None):
        """
        x: is the tensor of shape `[batch_size, channels, height, width]`
        """
        # Normalize
        x_norm = self.norm(x)

        # Get query, key and vector embeddings
        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)

        # Reshape to query, key and vector embeddings from
        # `[batch_size, channels, height, width]` to
        # `[batch_size, channels, height * width]`
        b, c, h, w = q.shape
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)

        # Compute $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)$
        attn = torch.einsum('bci, bcj->bij', q, k).contiguous() * self.scale
        attn = F.softmax(attn, dim=2)

        # Compute $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$
        out = torch.einsum('bij, bcj->bci', attn, v).contiguous()

        # Reshape back to `[batch_size, channels, height, width]`
        out = out.view(b, c, h, w)

        # Final convolution layer
        out = self.proj_out(out)

        # Add skip connection
        return x + out