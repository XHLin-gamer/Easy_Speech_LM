"""
テイラー展開: 1〜5項の近似を横一列（1×5）の正方形グラフで比較
各グラフに 元の関数 sin(x)（黒）と n 項の近似（色付き破線）を描く
"""
import math
import numpy as np
import matplotlib.pyplot as plt

# ---- 日本語フォント ----
plt.rcParams["font.family"] = "UD Digi Kyokasho N"


def taylor_sin(x, n_terms):
    """sin(x) のマクローリン展開を n_terms 項まで足す"""
    result = np.zeros_like(x)
    for k in range(n_terms):
        result += (-1) ** k * x ** (2 * k + 1) / math.factorial(2 * k + 1)
    return result


def term_label(n):
    """近似式の文字列 例: x - x^3/3! + x^5/5!"""
    parts = []
    for k in range(n):
        p = 2 * k + 1
        t = "x" if p == 1 else f"\\frac{{x^{p}}}{{{p}!}}"
        sign = "" if k == 0 else ("-" if k % 2 else "+")
        parts.append(sign + t)
    return "$" + "".join(parts) + "$"


x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
y_true = np.sin(x)
colors = plt.cm.viridis(np.linspace(0.15, 0.85, 5))

fig, axes = plt.subplots(1, 5, figsize=(20, 4.8))

for i, (ax, c) in enumerate(zip(axes, colors)):
    n = i + 1
    ax.plot(x, y_true, color="black", lw=2.5, label="元の関数 sin(x)")
    ax.plot(x, taylor_sin(x, n), color=c, lw=2.2, ls="--", label=f"{n}項の近似")
    ax.set_title(f"{n}項（{2 * n - 1}次まで）", fontsize=13)
    ax.text(0.5, 0.04, term_label(n), transform=ax.transAxes,
            ha="center", fontsize=16,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    ax.set_xlim(-2 * np.pi, 2 * np.pi)
    ax.set_ylim(-3, 3)
    ax.set_box_aspect(1)  # 正方形
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.grid(alpha=0.3)
    ax.set_xlabel("x")
    ax.legend(loc="upper left", fontsize=12)

axes[0].set_ylabel("y")
fig.suptitle("テイラー展開", fontsize=16)
fig.tight_layout()
fig.savefig("taylor_1x5.png", dpi=150)
# plt.show()