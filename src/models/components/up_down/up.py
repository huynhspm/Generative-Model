import torch
import torch.nn as nn
import torch.nn.functional as F

# Follow Stable Diffusion: https://nn.labml.ai/diffusion/stable_diffusion/model/unet.html
class UpSample(nn.Module):
    """
    ### Up-sampling layer
    """

    def __init__(self, channels: int) -> None:
        """
        channels: number of channels
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels=channels,
                              out_channels=channels,
                              kernel_size=3,
                              padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: input feature map: [batch_size, channels, height, width]

        """
        # Up-sample by a factor of 2
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # Apply convolution
        return self.conv(x)
    
if __name__ == "__main__":
    x = torch.randn(2, 1, 32, 32)
    up = UpSample(channels=1)
    out = up(x)

    print('***** UpSample *****')
    print('Input:', x.shape)
    print('Output:', out.shape)