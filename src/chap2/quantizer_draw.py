"""VectorQuantizer(8, 2) の潜在空間をボロノイ図で可視化する。

torch があれば元の VectorQuantizer をそのまま使い、無ければ NumPy で同じ計算
（uniform(INIT_LOW, INIT_HIGH) 初期化 → cdist → argmin）を再現する。
表示範囲は初期化区間の中心に合わせ、周囲に余白を付ける。
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from scipy.spatial import Voronoi

SEED = 2027
CODEBOOK_SIZE, LATENT_DIM = 8, 2
INIT_LOW, INIT_HIGH = 0.0, 1.0   # nn.init.uniform_ の区間
MARGIN = 0.15                     # 初期化区間の幅に対する余白の割合（片側）

# ---- 日本語フォント ----
plt.rcParams["font.family"] = "UD Digi Kyokasho N"

# ---- コードブックと入力点 ----
try:
    import torch
    import torch.nn as nn

    class VectorQuantizer(nn.Module):
        def __init__(self, codebook_size: int, latent_dim: int):
            super().__init__()
            self.embbeding = nn.Embedding(codebook_size, latent_dim)
            nn.init.uniform_(self.embbeding.weight, INIT_LOW, INIT_HIGH)

        def quantize(self, x):
            idx = torch.cdist(x, self.embbeding.weight).argmin(-1)
            z = self.embbeding.weight[idx]
            return z, idx

    torch.manual_seed(SEED)
    vq = VectorQuantizer(CODEBOOK_SIZE, LATENT_DIM)
    rand = torch.rand((1, 2))
    with torch.no_grad():
        z, idx = vq.quantize(rand)
        codebook = vq.embbeding.weight.numpy()
    x, z, idx = rand.numpy()[0], z.numpy()[0], int(idx[0])
    backend = "PyTorch"
except ImportError:
    rng = np.random.default_rng(SEED)
    codebook = rng.uniform(INIT_LOW, INIT_HIGH, (CODEBOOK_SIZE, LATENT_DIM))  # nn.init.uniform_
    x = rng.uniform(0, 1, LATENT_DIM)                             # torch.rand((1, 2))
    d = np.linalg.norm(x[None, :] - codebook, axis=-1)            # torch.cdist
    idx = int(d.argmin())                                         # argmin(-1)
    z = codebook[idx]                                             # embbeding.weight[idx]
    backend = "NumPy再現"

# ---- 描画 ----
center = (INIT_LOW + INIT_HIGH) / 2
half = (INIT_HIGH - INIT_LOW) / 2 + MARGIN * (INIT_HIGH - INIT_LOW)
lo, hi = center - half, center + half
n = 800
gx, gy = np.meshgrid(np.linspace(lo, hi, n), np.linspace(lo, hi, n))
grid = np.stack([gx.ravel(), gy.ravel()], -1)
region = np.linalg.norm(grid[:, None, :] - codebook[None], axis=-1).argmin(-1).reshape(n, n)

palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756",
           "#72B7B2", "#B279A2", "#EECA3B", "#9D755D"]
fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
ax.imshow(region, origin="lower", extent=(lo, hi, lo, hi),
          cmap=ListedColormap(palette), alpha=0.28, interpolation="nearest")

# ボロノイ境界（遠方に補助点を置いて無限辺も有限化して描く）
far = 100 * np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
vor = Voronoi(np.vstack([codebook, far]))
for (p, q), rv in zip(vor.ridge_points, vor.ridge_vertices):
    if p < CODEBOOK_SIZE and q < CODEBOOK_SIZE and -1 not in rv:
        seg = vor.vertices[rv]
        ax.plot(seg[:, 0], seg[:, 1], color="#333", lw=1.2)

# 初期化範囲（図の中央）と torch.rand の範囲 [0,1)^2
w = INIT_HIGH - INIT_LOW
ax.add_patch(plt.Rectangle((INIT_LOW, INIT_LOW), w, w, fill=False, ls="--", lw=1.2, ec="#555"))
# ax.text(INIT_LOW, INIT_HIGH + 0.02 * (hi - lo),
#         f"コードブック初期化範囲 U({INIT_LOW:g}, {INIT_HIGH:g})", fontsize=9, color="#555")
# ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ls=":", lw=1.2, ec="#c00"))
# ax.text(0, -0.05 * (hi - lo), "torch.rand の範囲 [0, 1)²", fontsize=9, color="#c00")

# コードベクトル
for k, (cx, cy) in enumerate(codebook):
    ax.scatter(cx, cy, s=160, color=palette[k], ec="black", lw=1.2, zorder=5)
    crowded = any(np.linalg.norm(codebook[j] - codebook[k]) < 0.2 for j in range(k + 1, CODEBOOK_SIZE))
    ax.annotate(f"e{k}", (cx, cy), xytext=(-22, -16) if crowded else (7, 7), textcoords="offset points",
                fontsize=11, weight="bold", zorder=6)

# 入力 x → 量子化 z
ax.scatter(*x, marker="*", s=320, color="white", ec="black", lw=1.2, zorder=7)
ax.annotate("x = rand", x, xytext=(8, -16), textcoords="offset points", fontsize=11)
ax.annotate("", xy=z, xytext=x, zorder=6,
            arrowprops=dict(arrowstyle="->", lw=1.8, color="black", ls="--"))
ax.scatter(*z, s=420, facecolor="none", ec="black", lw=2, zorder=6)

ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
ax.set_xlabel("潜在次元 1", fontsize=18); ax.set_ylabel("潜在次元 2", fontsize=16)
ax.set_title(f"潜在空間のボロノイ図（仮）", fontsize=18)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig("vq_voronoi.png")
print(backend, "idx =", idx, "x =", x, "z =", z)