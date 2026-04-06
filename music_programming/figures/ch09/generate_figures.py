"""第9章の図を生成するスクリプト

使い方: .venv/bin/python3 figures/ch09/generate_figures.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
# simple-sound-programming のパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'simple-sound-programming'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import japanize_matplotlib

from audio_lib import AudioSignal

OUTPUT_DIR = os.path.dirname(__file__)


def simple_delay_impulse(impulse, delay_sec, feedback, wet_level, sample_rate):
    """ディレイのインパルス応答を計算"""
    delay_samples = int(delay_sec * sample_rate)
    data = impulse
    output = np.zeros(len(data) + delay_samples * 8)
    buf = np.zeros(delay_samples)
    idx = 0

    for n in range(len(data)):
        delayed = buf[idx]
        buf[idx] = data[n] + feedback * delayed
        output[n] = (1.0 - wet_level) * data[n] + wet_level * delayed
        idx = (idx + 1) % delay_samples

    for n in range(len(data), len(output)):
        delayed = buf[idx]
        buf[idx] = feedback * delayed
        output[n] = wet_level * delayed
        idx = (idx + 1) % delay_samples

    return output


def simple_comb_filter_impulse(impulse, delay_sec, feedback, sample_rate):
    """コムフィルタのインパルス応答を計算"""
    delay_samples = int(delay_sec * sample_rate)
    data = impulse
    buf = np.zeros(delay_samples)
    idx = 0
    output = np.zeros_like(data)

    for n in range(len(data)):
        delayed = buf[idx]
        buf[idx] = data[n] + feedback * delayed
        output[n] = delayed
        idx = (idx + 1) % delay_samples

    return output


def fig1_delay_impulse():
    """図1: ディレイのインパルス応答"""
    sr = 44100
    impulse = np.zeros(sr)
    impulse[0] = 1.0
    output = simple_delay_impulse(impulse, 0.1, 0.6, 1.0, sr)

    plt.figure(figsize=(12, 4))
    t = np.arange(len(output)) / sr
    plt.plot(t, output)
    plt.title("ディレイのインパルス応答（遅延 0.1秒, feedback=0.6）")
    plt.xlabel("時間 (秒)")
    plt.ylabel("振幅")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_delay_impulse.png'), dpi=150)
    plt.close()
    print("fig1_delay_impulse.png を生成しました")


def fig2_comb_frequency():
    """図2: コムフィルタの周波数特性"""
    sr = 44100
    impulse = np.zeros(sr)
    impulse[0] = 1.0
    output = simple_comb_filter_impulse(impulse, 0.01, 0.7, sr)

    from scipy.fft import fft
    spectrum = np.abs(fft(output))
    freqs = np.arange(len(spectrum)) / len(spectrum) * sr

    plt.figure(figsize=(12, 4))
    plt.plot(freqs[:5000], 20 * np.log10(spectrum[:5000] + 1e-10))
    plt.title("コムフィルタの周波数特性")
    plt.xlabel("周波数 (Hz)")
    plt.ylabel("振幅 (dB)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_comb_frequency.png'), dpi=150)
    plt.close()
    print("fig2_comb_frequency.png を生成しました")


def fig4_compressor_waveform():
    """図4: コンプレッサーの波形比較"""
    from audio_lib import sine_wave, adsr
    from audio_lib.synthesis.note_utils import note_to_frequency

    sr = 44100
    notes = [60, 64, 67, 72, 67, 64]
    amplitudes = [0.3, 0.9, 0.5, 1.0, 0.4, 0.8]

    parts = []
    for midi_num, amp in zip(notes, amplitudes):
        freq = note_to_frequency(midi_num)
        wave = sine_wave(freq, 0.5)
        env = adsr(0.5, attack=0.01, decay=0.1, sustain=0.6, release=0.1)
        shaped = wave.data * env.data * amp
        parts.append(shaped)

    dynamic_data = np.concatenate(parts)

    # シンプルなコンプレッサー
    threshold = 0.4
    ratio = 4.0
    attack = 0.01
    release = 0.1
    envelope = 0.0
    attack_coeff = np.exp(-1.0 / (attack * sr))
    release_coeff = np.exp(-1.0 / (release * sr))
    compressed_data = np.zeros_like(dynamic_data)

    for n in range(len(dynamic_data)):
        current_level = abs(dynamic_data[n])
        if current_level > envelope:
            envelope += (current_level - envelope) * (1 - attack_coeff)
        else:
            envelope += (current_level - envelope) * (1 - release_coeff)

        if envelope > threshold:
            excess = envelope - threshold
            target = threshold + excess / ratio
            gain = target / envelope
        else:
            gain = 1.0
        compressed_data[n] = dynamic_data[n] * gain

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    t = np.arange(len(dynamic_data)) / sr

    axes[0].plot(t, dynamic_data, alpha=0.7)
    axes[0].set_title("元の音（音量バラバラ）")
    axes[0].set_ylabel("振幅")
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, compressed_data, alpha=0.7, color='orange')
    axes[1].set_title("コンプレッサー適用後")
    axes[1].set_xlabel("時間 (秒)")
    axes[1].set_ylabel("振幅")
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_compressor_waveform.png'),
                dpi=150)
    plt.close()
    print("fig4_compressor_waveform.png を生成しました")


def fig3_compressor_diagram():
    """図3: コンプレッサーの入出力特性図"""
    threshold = 0.5
    ratio = 4.0

    input_level = np.linspace(0, 1.0, 500)
    output_level = np.where(
        input_level <= threshold,
        input_level,
        threshold + (input_level - threshold) / ratio,
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    # 1:1 のリファレンス線（圧縮なし）
    ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5, label='圧縮なし (1:1)')

    # 圧縮カーブ
    ax.plot(input_level, output_level, linewidth=2.5, color='#1f77b4',
            label=f'圧縮あり ({ratio:.0f}:1)')

    # 閾値の補助線
    ax.axvline(threshold, color='red', linestyle=':', alpha=0.6)
    ax.axhline(threshold, color='red', linestyle=':', alpha=0.6)
    ax.plot(threshold, threshold, 'o', color='red', markersize=8, zorder=5)

    # ラベル
    ax.annotate('閾値 (threshold)',
                xy=(threshold, threshold),
                xytext=(threshold + 0.12, threshold - 0.1),
                fontsize=11, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

    # 圧縮される領域を薄く塗る
    ax.fill_between(input_level, output_level, input_level,
                    where=(input_level > threshold),
                    alpha=0.12, color='orange', label='圧縮される量')

    ax.set_xlabel('入力レベル', fontsize=12)
    ax.set_ylabel('出力レベル', fontsize=12)
    ax.set_title('コンプレッサーの入出力特性', fontsize=13)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_compressor_diagram.png'),
                dpi=150)
    plt.close()
    print("fig3_compressor_diagram.png を生成しました")


if __name__ == '__main__':
    fig1_delay_impulse()
    fig2_comb_frequency()
    fig3_compressor_diagram()
    fig4_compressor_waveform()
