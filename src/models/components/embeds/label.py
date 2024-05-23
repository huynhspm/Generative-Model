import torch
import torch.nn as nn


class LabelEmbedder(nn.Module):

    def __init__(self, n_classes: int, d_embed: int):
        super().__init__()

        self.label_embedding = nn.Embedding(num_embeddings=n_classes,
                                            embedding_dim=d_embed)

    def forward(self, label: torch.Tensor):
        return self.label_embedding(label)


if __name__ == "__main__":
    x = torch.randint(0, 10, (2, ))
    labelEmbedder = LabelEmbedder(n_classes=10, d_embed=64)
    out = labelEmbedder(x)

    print('***** LabelEmbedder *****')
    print('Input:', x.shape)
    print('Output:', out.shape)
