"""第11章の図を生成するスクリプト"""

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401

import sys
sys.path.insert(0, "/Users/gsuzuki/projects/writing/simple-sound-programming")


def fig1_note_values():
    """fig1: 音符の種類と拍数の関係（概念図）"""
    fig, ax = plt.subplots(figsize=(9, 3.5))

    notes = [
        ("全音符",   4,    "#1565C0"),
        ("2分音符",  2,    "#1E88E5"),
        ("4分音符",  1,    "#42A5F5"),
        ("8分音符",  0.5,  "#90CAF9"),
        ("16分音符", 0.25, "#BBDEFB"),
    ]

    for i, (name, beats, color) in enumerate(notes):
        ax.barh(i, beats, height=0.6, color=color,
                edgecolor="white", linewidth=1.5)
        if beats >= 0.5:
            ax.text(beats / 2, i, f"{name}\n{beats}拍",
                    ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white")
        else:
            ax.text(beats + 0.08, i, f"{name}  {beats}拍",
                    ha="left", va="center", fontsize=11,
                    fontweight="bold", color=color)

    for beat in range(1, 5):
        ax.axvline(x=beat, color="#BDBDBD", linewidth=0.8, linestyle="--")

    ax.set_xlim(0, 4.5)
    ax.set_ylim(-0.5, len(notes) + 0.3)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["0", "1", "2", "3", "4"], fontsize=11)
    ax.set_xlabel("拍数", fontsize=12)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_title("音符の種類と長さ（4/4拍子）", fontsize=13)

    ax.annotate("", xy=(4, len(notes) + 0.05), xytext=(0, len(notes) + 0.05),
                arrowprops=dict(arrowstyle="<->", color="#616161", lw=1.5))
    ax.text(2, len(notes) + 0.15, "1小節 = 4拍", ha="center", va="top",
            fontsize=10, color="#616161")

    plt.tight_layout()
    plt.savefig("figures/ch11/fig1_note_values.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("fig1_note_values.png を生成しました")


def fig2_drum_pattern():
    """fig2: 8ビートのドラムパターンのグリッド表示"""
    patterns = {
        "ハイハット": [1, 1, 1, 1, 1, 1, 1, 1],
        "スネア":     [0, 0, 1, 0, 0, 0, 1, 0],
        "キック":     [1, 0, 0, 0, 1, 0, 0, 0],
    }

    fig, ax = plt.subplots(figsize=(8, 2.5))

    instruments = list(patterns.keys())
    n_steps = 8

    for row, (name, pattern) in enumerate(patterns.items()):
        for step, hit in enumerate(pattern):
            color = "#2196F3" if hit else "#E0E0E0"
            rect = plt.Rectangle((step, row), 0.9, 0.8,
                                 facecolor=color, edgecolor="white",
                                 linewidth=2)
            ax.add_patch(rect)

    ax.set_xlim(-0.1, n_steps + 0.1)
    ax.set_ylim(-0.2, len(instruments) + 0.2)
    ax.set_xticks([i + 0.45 for i in range(n_steps)])
    ax.set_xticklabels([str(i + 1) for i in range(n_steps)], fontsize=11)
    ax.set_yticks([i + 0.4 for i in range(len(instruments))])
    ax.set_yticklabels(instruments, fontsize=12)
    ax.set_xlabel("ステップ（8分音符単位）", fontsize=11)
    ax.set_title("8ビートの基本パターン", fontsize=13)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # 小節線（4ステップ目の後）
    ax.axvline(x=4, color="gray", linewidth=1.5, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("figures/ch11/fig2_drum_pattern.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("fig2_drum_pattern.png を生成しました")


def fig3_score_to_waveform():
    """fig3: スコアデータから波形が生成される様子"""
    from audio_lib.synthesis.oscillators import sine_wave
    from audio_lib.synthesis.envelopes import adsr
    from audio_lib.synthesis.note_utils import note_name_to_number, note_to_frequency

    score = [
        ("C4", 1), ("E4", 1), ("G4", 1), ("C5", 2),
    ]

    bpm = 120
    sr = 44100

    total_beats = sum(b for _, b in score)
    total_sec = total_beats * 60.0 / bpm
    total_samples = int(sr * total_sec)
    output = np.zeros(total_samples)
    t_full = np.linspace(0, total_sec, total_samples, endpoint=False)

    # 各音の区間情報を記録
    note_regions = []
    current = 0.0
    for note_name, beats in score:
        dur = beats * 60.0 / bpm
        if note_name is not None:
            nn = note_name_to_number(note_name)
            freq = note_to_frequency(nn)
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)
            wave = np.sin(2 * np.pi * freq * t)
            env = adsr(dur, attack=0.01, decay=0.05, sustain=0.8,
                       release=0.05, sample_rate=sr)
            wave = wave * env.data
            start = int(current * sr)
            end = start + len(wave)
            output[start:end] += wave
            note_regions.append((current, current + dur, note_name))
        current += dur

    # 正規化
    mx = np.max(np.abs(output))
    if mx > 0:
        output = output * 0.8 / mx

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.5),
                                    gridspec_kw={"height_ratios": [1, 2.5]})

    # 上段: スコアの可視化
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    for i, (start, end, name) in enumerate(note_regions):
        ax1.barh(0, end - start, left=start, height=0.6,
                 color=colors[i % len(colors)], edgecolor="white",
                 linewidth=1.5)
        ax1.text((start + end) / 2, 0, name, ha="center", va="center",
                 fontsize=11, fontweight="bold", color="white")

    ax1.set_xlim(0, total_sec)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_yticks([])
    ax1.set_title("スコアデータ → 波形", fontsize=13)
    ax1.set_xlabel("")
    ax1.set_xticks([])

    # 下段: 波形
    ax2.plot(t_full, output, color="#333333", linewidth=0.3)
    for i, (start, end, name) in enumerate(note_regions):
        ax2.axvspan(start, end, alpha=0.1, color=colors[i % len(colors)])
    ax2.set_xlim(0, total_sec)
    ax2.set_ylim(-1, 1)
    ax2.set_xlabel("時間（秒）", fontsize=11)
    ax2.set_ylabel("振幅", fontsize=11)

    plt.tight_layout()
    plt.savefig("figures/ch11/fig3_score_to_waveform.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("fig3_score_to_waveform.png を生成しました")


if __name__ == "__main__":
    fig1_note_values()
    fig2_drum_pattern()
    fig3_score_to_waveform()
