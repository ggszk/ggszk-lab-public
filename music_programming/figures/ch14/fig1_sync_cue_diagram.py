"""
fig1_sync_cue_diagram.py
sync / cue によるループ同期の仕組みを示すタイミングチャート

14.2 節の metronome + kick + hihat の例に対応:
  :metronome が cue :tick を送り、:kick と :hihat が sync :tick で受け取る
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import japanize_matplotlib  # noqa: F401

# --- パラメータ ---
bpm = 120
beat_sec = 60.0 / bpm  # 0.5秒
n_beats = 8
total_time = n_beats * beat_sec

# ループ定義（名前, Y位置, イベント）
# イベント: (時刻, 種類)  種類='cue'|'sound'
loops = [
    {
        "name": ":metronome",
        "y": 3,
        "color": "#5B9BD5",
        "events": [(i * beat_sec, "cue") for i in range(n_beats)],
        "label": "cue :tick",
    },
    {
        "name": ":kick",
        "y": 2,
        "color": "#ED7D31",
        "events": [(i * beat_sec, "sound") for i in range(n_beats)],
        "label": "sample :bd_haus",
    },
    {
        "name": ":hihat",
        "y": 1,
        "color": "#70AD47",
        "events": [(i * beat_sec, "sound") for i in range(n_beats)],
        "label": "sample :hat_snap",
    },
]

fig, ax = plt.subplots(figsize=(12, 4.5))

# --- タイムライン描画 ---
for loop in loops:
    y = loop["y"]
    ax.annotate(
        loop["name"],
        xy=(-0.15, y),
        fontsize=13,
        fontfamily="monospace",
        fontweight="bold",
        color=loop["color"],
        ha="right",
        va="center",
    )
    # 横線（タイムライン）
    ax.plot(
        [0, total_time],
        [y, y],
        color=loop["color"],
        linewidth=1.5,
        alpha=0.4,
    )

# --- イベントマーカー ---
for loop in loops:
    y = loop["y"]
    for t, kind in loop["events"]:
        if kind == "cue":
            # cue: 下向き三角
            ax.plot(t, y, marker="v", markersize=10, color=loop["color"], zorder=5)
        else:
            # sound: 丸
            ax.plot(t, y, marker="o", markersize=8, color=loop["color"], zorder=5)

# --- sync の矢印（cue → sync） ---
for i in range(n_beats):
    t = i * beat_sec
    metro_y = 3
    for target_y in [2, 1]:
        ax.annotate(
            "",
            xy=(t, target_y + 0.12),
            xytext=(t, metro_y - 0.12),
            arrowprops=dict(
                arrowstyle="->",
                color="#888888",
                lw=1.0,
                linestyle="--",
                connectionstyle="arc3,rad=0",
            ),
        )

# --- 拍番号 ---
for i in range(n_beats):
    t = i * beat_sec
    ax.text(t, 3.55, f"{i+1}", ha="center", va="bottom", fontsize=10, color="#555555")
ax.text(total_time / 2, 3.9, "拍", ha="center", va="bottom", fontsize=11, color="#555555")

# --- 凡例 ---
legend_elements = [
    mpatches.Patch(facecolor="none", edgecolor="none",
                   label="▼  cue :tick（合図を送る）"),
    mpatches.Patch(facecolor="none", edgecolor="none",
                   label="●  sync :tick で受け取り → 音を鳴らす"),
    mpatches.Patch(facecolor="none", edgecolor="none",
                   label="- -  同期の流れ"),
]
ax.legend(
    handles=legend_elements,
    loc="lower right",
    fontsize=10,
    framealpha=0.8,
    edgecolor="#cccccc",
)

# --- 軸設定 ---
ax.set_xlim(-0.2, total_time + 0.1)
ax.set_ylim(0.4, 4.2)
ax.set_xlabel("時間（秒）", fontsize=11)
ax.set_xticks([i * beat_sec for i in range(n_beats + 1)])
ax.set_xticklabels([f"{i * beat_sec:.1f}" for i in range(n_beats + 1)])
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.grid(axis="x", alpha=0.2)

ax.set_title("sync / cue によるループ同期の仕組み", fontsize=14, pad=20)

plt.tight_layout()
plt.savefig("figures/ch14/fig1_sync_cue_diagram.png", dpi=150, bbox_inches="tight")
plt.close()
print("生成完了: figures/ch14/fig1_sync_cue_diagram.png")
