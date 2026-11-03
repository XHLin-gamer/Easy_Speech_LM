import os
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")  # 画面なしでも保存できるように
import matplotlib.pyplot as plt

from vqvae import VQVAE

device = "cuda" if torch.cuda.is_available() else "cpu"
tf = transforms.ToTensor()
train = datasets.MNIST("data", train=True, download=True, transform=tf)
test = datasets.MNIST("data", train=False, download=True, transform=tf)
loader = DataLoader(train, batch_size=128, shuffle=True)

model = VQVAE().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

EPOCHS, LOG_EVERY = 10, 50
steps_per_epoch = len(loader)

# 毎エポック同じ画像で比較する：テストセットから 0〜9 を 1 枚ずつ
sample_ids = [(test.targets == d).nonzero()[0].item() for d in range(10)]
sample_x = torch.stack([test[i][0] for i in sample_ids]).to(device)
os.makedirs("samples", exist_ok=True)


@torch.no_grad()
def save_samples(model, x, epoch, path):
    was_training = model.training
    model.eval()
    x_hat, _ = model(x)
    model.train(was_training)

    x, x_hat = x.cpu().squeeze(1), x_hat.cpu().squeeze(1)
    mse = F.mse_loss(x_hat, x).item()
    n = len(x)

    fig, axes = plt.subplots(2, n, figsize=(n * 1.0, 2.4))
    for i in range(n):
        for row, img in enumerate((x[i], x_hat[i])):
            ax = axes[row, i]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Recon", fontsize=10)
    fig.suptitle(f"Epoch {epoch}   test MSE {mse:.4f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


save_samples(model, sample_x, 0, "samples/epoch_00.png")  # 学習前

for epoch in range(1, EPOCHS + 1):
    model.train()
    total = run_recon = run_vq = 0.0
    t0 = time.time()

    for step, (x, _) in enumerate(loader, 1):
        x = x.to(device)
        x_hat, vq_loss = model(x)
        recon = F.binary_cross_entropy(x_hat, x)
        loss = recon + vq_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item()
        run_recon += recon.item()
        run_vq += vq_loss.item()

        if step % LOG_EVERY == 0 or step == steps_per_epoch:
            n = LOG_EVERY if step % LOG_EVERY == 0 else step % LOG_EVERY
            print(f"epoch {epoch}/{EPOCHS}  step {step:4d}/{steps_per_epoch}  "
                  f"recon {run_recon/n:.4f}  vq {run_vq/n:.4f}  "
                  f"{time.time()-t0:.1f}s", flush=True)
            run_recon = run_vq = 0.0

    path = f"samples/epoch_{epoch:02d}.png"
    save_samples(model, sample_x, epoch, path)
    print(f"== epoch {epoch} done  avg loss {total/steps_per_epoch:.4f}  "
          f"({time.time()-t0:.1f}s)  -> {path}", flush=True)

torch.save(model.state_dict(), "vqvae_mnist.pt")
print("saved to vqvae_mnist.pt")