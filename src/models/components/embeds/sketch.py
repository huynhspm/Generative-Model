import torch
import torch.nn as nn

class SketchEmbedding(nn.Module):
    def __init__(self,  
                 in_channels: int, 
                 d_cond: int):
        super().__init__()

        self.sketch_embedding = nn.Conv2d(in_channels=in_channels, 
                                          out_channels=d_cond,
                                          kernel_size=3,
                                          padding=1)
    def forward(self, sketch: torch.Tensor):
        out = self.sketch_embedding(sketch)
        batch_size, channels, _, _ = out.shape
        return out.reshape(batch_size, channels, -1).swapaxes(1, 2)

if __name__ == "__main__":
    x = torch.randn(2, 3, 64, 64)
    sketchEmbedding = SketchEmbedding(in_channels=3, d_cond=64)
    out = sketchEmbedding(x)

    print('***** SketchEmbedding *****')
    print('Input:', x.shape)
    print('Output:', out.shape)
