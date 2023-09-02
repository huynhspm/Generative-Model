import torch
import torch.nn as nn

class LabelEmbedding(nn.Module):
    def __init__(self,  
                 n_classes: int, 
                 d_cond: int):
        super().__init__()

        self.label_embedding = nn.Sequential(nn.Embedding(num_embeddings=n_classes, 
                                                          embedding_dim=d_cond),
                                             nn.SiLU(),
                                             nn.Linear(in_features=d_cond, 
                                                       out_features=d_cond))
        
    def forward(self, label: torch.Tensor):
        return self.label_embedding(label)


if __name__ == "__main__":
    x = torch.randint(0, 10, (2,))
    labelEmbedding = LabelEmbedding(n_classes=10, d_cond=64)
    out = labelEmbedding(x)

    print('***** LabelEmbedding *****')
    print('Input:', x.shape)
    print('Output:', out.shape)
