import math
import torch
import torch.nn as nn

class TimeEmbedder(nn.Module):
    def __init__(self,  
                 d_model: int, 
                 dim: int, 
                 n_steps: int = 1000,
                 max_period: int = 10000):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(max_period)
        emb = torch.exp(-emb)
        pos = torch.arange(n_steps).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [n_steps, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [n_steps, d_model // 2, 2]
        emb = emb.view(n_steps, d_model)

        self.time_embedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, time_steps):
        return self.time_embedding(time_steps)
    
if __name__ == '__main__':

    timeEmbedder = TimeEmbedder(d_model=10, dim=64)
    timesteps = torch.arange(0, 1000)
    te = timeEmbedder(timesteps)
    print(te.shape)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    plt.plot(timesteps, te[:, [10, 20, 40, 60]].detach().numpy())
    plt.legend(["dim %d" % p for p in [10, 20, 40, 60]])
    plt.title("Time embeddings")
    plt.show()