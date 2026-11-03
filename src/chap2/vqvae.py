
import torch
from torch import nn
from torch.nn import functional as F
torch.manual_seed(42)


# DIM   : 潜在ベクトルの次元数
# CODES : コードブックのサイズ（コード数）
# BETA  : Commitment Loss の重み
DIM, CODES, BETA = 4, 64, 0.25

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, latent_dim: int):
        super().__init__()

        self.embbeding = nn.Embedding(codebook_size, latent_dim)
        nn.init.uniform_(self.embbeding.weight, 0, 1)

    def quantize(self, x):
        idx = torch.cdist(x, self.embbeding.weight).argmin(-1) # argminで勾配が消失しまう
        z = self.embbeding(idx)
        # z_q = x + (z - x).detach()    
        return z
    
    def forward(self, x):
        return self.quantize(x)

class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        # kernel = stride = 2 → 畳み込みの窓が重ならない
        # 2 層で縦横 1/4 になる（28 → 7）。潜在ベクトル 1 つ => 4×4 ピクセルのパッチ 1 つ
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(in_channels=32, out_channels=DIM, kernel_size=2, stride=2)
        )

        # エンコーダと対称な構成（7 → 28 に戻す）
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(in_channels=DIM, out_channels=32, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

        # コードブック：CODES 個の DIM 次元ベクトル
        self.codebook = nn.Embedding(CODES, DIM)
        nn.init.uniform_(self.codebook.weight, -1 / CODES, 1 / CODES)

    def encode(self, x):
        """
        x: (B, 1, 28, 28)
        戻り値: z   (B*7*7, DIM) 潜在ベクトル
                idx (B*7*7,)     各ベクトルに最も近いコード番号
        """
        z_raw = self.enc(x)  # (B, DIM, 7, 7)

        # チャネル（＝潜在次元）を最後に移し、位置ごとのベクトルをすべて一行に並べる
        z = z_raw.permute(0, 2, 3, 1).reshape(-1, DIM)
        
        # 全コードとのユークリッド距離を計算し、最も近いコードを選ぶ
        return z, torch.cdist(z, self.codebook.weight).argmin(1)

    def forward(self, x):
        z, idx = self.encode(x)
        q = self.codebook(idx)  # 量子化後のベクトル

        # 第1項: コードブックをエンコーダ出力に近づける（エンコーダ側は固定）
        # 第2項: エンコーダ出力をコードに近づける＝Commitment Loss（コードブック側は固定）
        codebook_loss = F.mse_loss(q, z.detach())
        commitment_loss = F.mse_loss(z, q.detach())
        vq_loss = codebook_loss + BETA * commitment_loss

        # Straight-Through 推定：順伝播では q を使い、逆伝播では勾配を z にそのまま流す（argmin は微分できないため）
        q = z + (q - z).detach()

        # (B*7*7, DIM) → (B, DIM, 7, 7) に戻してデコーダへ
        q = q.view(len(x), 7, 7, DIM).permute(0, 3, 1, 2)
        return self.dec(q), vq_loss
    
    
    