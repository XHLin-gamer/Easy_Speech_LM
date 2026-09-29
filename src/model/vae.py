import torch
from torch import nn
import torch.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_sise: int = 1024, latent_dim = 32):
        super().__init__()

        self.emb = nn.Embedding(codebook_sise, latent_dim)
        nn.init.uniform_(self.emb.weight, 0., 1.)

    def forward(self, x):
        # (L-C) ** 2 = L**2 - 2LC + C**2
        # x.shape -> [B, D]; emb.weight.shape -> [N, D]
        L2 = torch.sum(self.emb.weight ** 2, dim=-1).unsqueeze(0)                  # [1, N]
        C2 = torch.sum(x ** 2, dim=-1).unsqueeze(1)                                # [B, 1]
        LC = torch.sum(x @ self.emb.weight.t(), dim=-1, keepdim=True)              # [B, N]
        # print(L2.shape, LC.shape, C2.shape)
        distences = L2 - 2 * LC + C2
        min_distence_id = torch.argmin(distences, dim=-1)
        e = self.emb.weight[min_distence_id]
        return e


VQ = VectorQuantizer(8, 1)
rand = torch.rand((4, 1))
print(VQ(rand).shape)
