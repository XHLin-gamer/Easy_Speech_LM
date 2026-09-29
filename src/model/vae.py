import torch
from torch import nn
from torch import Tensor
import torch.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_sise: int = 1024, latent_dim = 32):
        super().__init__()

        self.emb = nn.Embedding(codebook_sise, latent_dim)
        nn.init.uniform_(self.emb.weight, 0., 1.)

    def forward(self, x: Tensor, efficient = True):
        # x.shape -> [B, D]; emb.weight.shape -> [N, D]
        if not efficient:
            # x.shape -> [B, 1, D]
            # emb.shape -> [1, N, D]
            _x = x.unsqueeze(1)
            _emb = self.emb.weight.unsqueeze(0)
            distences = torch.sum((_x - _emb) ** 2, dim = -1)

        else:
            # (L-C) ** 2 = L**2 - 2LC + C**2
            L2 = torch.sum(self.emb.weight ** 2, dim=-1).unsqueeze(0)                  # [1, N]
            C2 = torch.sum(x ** 2, dim=-1).unsqueeze(1)                                # [B, 1]
            LC = torch.sum(x @ self.emb.weight.t(), dim=-1, keepdim=True)              # [B, N]
            # print(L2.shape, LC.shape, C2.shape)
            distences = L2 - 2 * LC + C2
        min_distence_id = torch.argmin(distences, dim=-1)
        e = self.emb.weight[min_distence_id]
        e = x + (e - x).detach()
        return e



VQ = VectorQuantizer(8, 1)
rand = torch.rand((4, 1))
print(VQ(rand, efficient = False).shape)
