"""第12章の図を生成するスクリプト"""

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401


def fig1_8beat_pattern():
    """fig1: 8ビートの基本パターン（Sonic Pi版）
    ch11の fig2 と同じグリッドだが、Sonic Pi のコード表記を添える。
    """
    patterns = {
        "ハイハット\n:hat_snap": [1, 1, 1, 1, 1, 1, 1, 1],
        "スネア\n:sn_dub":      [0, 0, 1, 0, 0, 0, 1, 0],
        "キック\n:bd_haus":     [1, 0, 0, 0, 1, 0, 0, 0],
    }

    fig, ax = plt.subplots(figsize=(8, 3))

    instruments = list(patterns.keys())
    n_steps = 8

    for row, (name, pattern) in enumerate(patterns.items()):
        for step, hit in enumerate(pattern):
            color = "#2196F3" if hit else "#E0E0E0"
            rect = plt.Rectangle((step, row), 0.9, 0.8,
                                 facecolor=color, edgecolor="white",
                                 linewidth=2)
            ax.add_patch(rect)
            if hit:
                ax.text(step + 0.45, row + 0.4, "1",
                        ha="center", va="center", fontsize=10,
                        fontweight="bold", color="white")
            else:
                ax.text(step + 0.45, row + 0.4, "0",
                        ha="center", va="center", fontsize=10,
                        color="#9E9E9E")

    ax.set_xlim(-0.1, n_steps + 0.1)
    ax.set_ylim(-0.2, len(instruments) + 0.2)
    ax.set_xticks([i + 0.45 for i in range(n_steps)])
    ax.set_xticklabels([str(i + 1) for i in range(n_steps)], fontsize=11)
    ax.set_yticks([i + 0.4 for i in range(len(instruments))])
    ax.set_yticklabels(instruments, fontsize=11)
    ax.set_xlabel("ステップ（8分音符単位）", fontsize=11)
    ax.set_title("8ビートの基本パターン（Sonic Pi）", fontsize=13)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # 小節線
    ax.axvline(x=4, color="gray", linewidth=1.5, linestyle="--", alpha=0.5)

    # 拍番号
    for beat in range(4):
        ax.text(beat * 2 + 0.95, len(instruments) + 0.05,
                f"{beat + 1}拍目", ha="center", va="top",
                fontsize=9, color="#757575")

    plt.tight_layout()
    plt.savefig("figures/ch12/fig1_8beat_pattern.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("fig1_8beat_pattern.png を生成しました")


def fig2_spread_euclidean():
    """fig2: spread(3, 8) のユークリッドリズムを円環上に可視化"""
    n_total = 8
    n_hits = 3

    # spread(3, 8) の結果: true の位置
    hit_positions = [0, 3, 5]  # (ring true, false, false, true, false, true, false, false)

    fig, ax = plt.subplots(figsize=(5, 5))

    # 円を描く
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n_total,
                         endpoint=False)
    radius = 1.5

    # ステップの円
    for i in range(n_total):
        x = radius * np.cos(angles[i])
        y = radius * np.sin(angles[i])

        if i in hit_positions:
            circle = plt.Circle((x, y), 0.18, facecolor="#2196F3",
                                edgecolor="white", linewidth=2, zorder=3)
            ax.text(x, y, "●", ha="center", va="center",
                    fontsize=14, color="white", fontweight="bold", zorder=4)
        else:
            circle = plt.Circle((x, y), 0.18, facecolor="#E0E0E0",
                                edgecolor="white", linewidth=2, zorder=3)
            ax.text(x, y, "○", ha="center", va="center",
                    fontsize=14, color="#9E9E9E", zorder=4)
        ax.add_patch(circle)

        # ステップ番号（外側）
        label_r = radius + 0.35
        lx = label_r * np.cos(angles[i])
        ly = label_r * np.sin(angles[i])
        ax.text(lx, ly, str(i + 1), ha="center", va="center",
                fontsize=11, color="#616161")

    # 打点を結ぶ線
    for i in range(len(hit_positions)):
        j = (i + 1) % len(hit_positions)
        x1 = radius * np.cos(angles[hit_positions[i]])
        y1 = radius * np.sin(angles[hit_positions[i]])
        x2 = radius * np.cos(angles[hit_positions[j]])
        y2 = radius * np.sin(angles[hit_positions[j]])
        ax.plot([x1, x2], [y1, y2], color="#2196F3", linewidth=1.5,
                alpha=0.4, zorder=2)

    # 中央のラベル
    ax.text(0, 0.15, "spread(3, 8)", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#333333")
    ax.text(0, -0.2, "8ステップに3打点", ha="center", va="center",
            fontsize=10, color="#757575")

    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-2.3, 2.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("ユークリッドリズム spread(3, 8)", fontsize=13, pad=15)

    plt.tight_layout()
    plt.savefig("figures/ch12/fig2_spread_euclidean.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("fig2_spread_euclidean.png を生成しました")


if __name__ == "__main__":
    fig1_8beat_pattern()
    fig2_spread_euclidean()
