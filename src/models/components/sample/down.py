import torch
import torch.nn as nn

class DownSample(nn.Module):
    """
    ### Down-sampling layer
    """

    def __init__(self, channels: int) -> None:
        """
        channels: is the number of channels
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels=channels,
                              out_channels=channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Apply convolution
        return self.conv(x)
    
if __name__ == "__main__":
    x = torch.randn(2, 1, 32, 32)
    up = DownSample(channels=1)
    out = up(x)

    print('***** DownSample *****')
    print('Input:', x.shape)
    print('Output:', out.shape)