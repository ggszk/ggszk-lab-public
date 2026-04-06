"""第13回「音を周波数で見る」の図を生成するスクリプト"""

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401

import sys
sys.path.insert(0, "/Users/gsuzuki/projects/writing/simple-sound-programming")
from audio_lib import (
    sine_wave, sawtooth_wave, square_wave, triangle_wave,
    additive_synth, adsr, AudioSignal,
    note_to_frequency,
)
from audio_lib.effects.audio_effects import Reverb

SAMPLE_RATE = 44100
OUTPUT_DIR = "/Users/gsuzuki/projects/writing/music-programming-book/figures/ch13"


def fig1_simple_spectrum():
    """fig1: 合成音のスペクトル（440 + 880 + 1320 Hz）"""
    duration = 1.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    data = (np.sin(2 * np.pi * 440 * t)
            + 0.5 * np.sin(2 * np.pi * 880 * t)
            + 0.3 * np.sin(2 * np.pi * 1320 * t))

    fft_result = np.fft.fft(data)
    fft_freq = np.fft.fftfreq(len(data), 1 / SAMPLE_RATE)
    fft_magnitude = np.abs(fft_result)

    positive = fft_freq >= 0
    freq = fft_freq[positive]
    magnitude = fft_magnitude[positive]
    range_idx = freq <= 3000

    plt.figure(figsize=(12, 4))
    plt.plot(freq[range_idx], magnitude[range_idx])
    plt.title("合成音のスペクトル（440Hz + 880Hz + 1320Hz）")
    plt.xlabel("周波数 (Hz)")
    plt.ylabel("振幅")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig1_simple_spectrum.png", dpi=150)
    plt.close()
    print("fig1 done")


def fig2_four_waveforms_spectrum():
    """fig2: 4つの基本波形のスペクトル比較"""
    f0 = 440
    dur = 1.0

    waves = {
        "サイン波": sine_wave(f0, dur),
        "ノコギリ波": sawtooth_wave(f0, dur),
        "矩形波": square_wave(f0, dur),
        "三角波": triangle_wave(f0, dur),
    }

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    for ax, (name, sig) in zip(axes, waves.items()):
        data = sig.data
        fft_mag = np.abs(np.fft.fft(data))
        freqs = np.fft.fftfreq(len(data), 1 / sig.sample_rate)
        pos = freqs >= 0
        f = freqs[pos]
        m = fft_mag[pos]
        rng = f <= 5000

        ax.plot(f[rng], 20 * np.log10(m[rng] + 1e-10))
        ax.set_ylabel(f"{name}\n振幅 (dB)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("周波数 (Hz)")
    fig.suptitle("4つの基本波形のスペクトル比較", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig2_four_waveforms_spectrum.png", dpi=150)
    plt.close()
    print("fig2 done")


def fig3_window_problem():
    """fig3: 窓関数なしの比較（スペクトル漏れ）"""
    dur = 0.1
    t_short = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)

    data_440 = np.sin(2 * np.pi * 440 * t_short)
    data_443 = np.sin(2 * np.pi * 443 * t_short)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, (data, freq_label) in zip(axes, [(data_440, "440Hz"), (data_443, "443Hz")]):
        mag = np.abs(np.fft.fft(data))
        freqs = np.fft.fftfreq(len(data), 1 / SAMPLE_RATE)
        pos = freqs >= 0
        rng = freqs[pos] <= 600
        ax.plot(freqs[pos][rng], mag[pos][rng])
        ax.set_title(f"{freq_label}（窓関数なし）")
        ax.set_xlabel("周波数 (Hz)")
        ax.set_ylabel("振幅")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig3_window_problem.png", dpi=150)
    plt.close()
    print("fig3 done")


def fig4_window_solved():
    """fig4: 窓関数ありの比較"""
    dur = 0.1
    t_short = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)

    data_440 = np.sin(2 * np.pi * 440 * t_short)
    data_443 = np.sin(2 * np.pi * 443 * t_short)
    window = np.hanning(len(t_short))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, (data, freq_label) in zip(axes, [(data_440, "440Hz"), (data_443, "443Hz")]):
        windowed = data * window
        mag = np.abs(np.fft.fft(windowed))
        freqs = np.fft.fftfreq(len(data), 1 / SAMPLE_RATE)
        pos = freqs >= 0
        rng = freqs[pos] <= 600
        ax.plot(freqs[pos][rng], mag[pos][rng])
        ax.set_title(f"{freq_label}（ハニング窓あり）")
        ax.set_xlabel("周波数 (Hz)")
        ax.set_ylabel("振幅")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig4_window_solved.png", dpi=150)
    plt.close()
    print("fig4 done")


def fig5_window_functions():
    """fig5: 代表的な窓関数"""
    n = 256
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.ones(n), label="矩形窓（窓なし）", alpha=0.7)
    ax.plot(np.hanning(n), label="ハニング窓", alpha=0.7)
    ax.plot(np.hamming(n), label="ハミング窓", alpha=0.7)
    ax.plot(np.blackman(n), label="ブラックマン窓", alpha=0.7)
    ax.set_title("代表的な窓関数")
    ax.set_xlabel("サンプル")
    ax.set_ylabel("振幅")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig5_window_functions.png", dpi=150)
    plt.close()
    print("fig5 done")


def fig6_chirp_spectrogram():
    """fig6: 上昇音のスペクトログラム"""
    dur = 2.0
    t_chirp = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    freq_start, freq_end = 220, 880
    instantaneous_freq = freq_start + (freq_end - freq_start) * t_chirp / dur
    phase = 2 * np.pi * np.cumsum(instantaneous_freq) / SAMPLE_RATE
    chirp = np.sin(phase)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(t_chirp[:4410], chirp[:4410])
    axes[0].set_title("波形（先頭 0.1 秒）")
    axes[0].set_xlabel("時間 (秒)")
    axes[0].set_ylabel("振幅")
    axes[0].grid(True, alpha=0.3)

    axes[1].specgram(chirp, Fs=SAMPLE_RATE, NFFT=1024, noverlap=512, cmap='magma')
    axes[1].set_title("スペクトログラム")
    axes[1].set_xlabel("時間 (秒)")
    axes[1].set_ylabel("周波数 (Hz)")
    axes[1].set_ylim(0, 2000)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig6_chirp_spectrogram.png", dpi=150)
    plt.close()
    print("fig6 done")


def fig7_piano_analysis():
    """fig7: ピアノ風の音のスペクトルとスペクトログラム"""
    f0 = 262
    dur = 2.0

    harmonics = {1: 1.0, 2: 0.7, 3: 0.3, 4: 0.2, 5: 0.1, 6: 0.05}
    wave = additive_synth(f0, harmonics, duration=dur)
    env = adsr(dur, attack=0.01, decay=0.4, sustain=0.3, release=0.5)
    piano_data = wave.data * env.data

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # スペクトル
    windowed = piano_data * np.hanning(len(piano_data))
    fft_mag = np.abs(np.fft.fft(windowed))
    freqs = np.fft.fftfreq(len(windowed), 1 / SAMPLE_RATE)
    pos = freqs >= 0
    rng = freqs[pos] <= 3000

    axes[0].plot(freqs[pos][rng], 20 * np.log10(fft_mag[pos][rng] + 1e-10))
    axes[0].set_title("ピアノ風の音のスペクトル")
    axes[0].set_xlabel("周波数 (Hz)")
    axes[0].set_ylabel("振幅 (dB)")
    axes[0].grid(True, alpha=0.3)

    # スペクトログラム
    axes[1].specgram(piano_data, Fs=SAMPLE_RATE, NFFT=2048, noverlap=1536, cmap='magma')
    axes[1].set_title("ピアノ風の音のスペクトログラム")
    axes[1].set_xlabel("時間 (秒)")
    axes[1].set_ylabel("周波数 (Hz)")
    axes[1].set_ylim(0, 3000)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig7_piano_analysis.png", dpi=150)
    plt.close()
    print("fig7 done")


def fig8_square_vs_saw_spectrogram():
    """fig8: 矩形波とノコギリ波のスペクトログラム比較"""
    f0 = 262
    dur = 1.5

    sq = square_wave(f0, dur)
    saw = sawtooth_wave(f0, dur)
    env = adsr(dur, attack=0.01, decay=0.2, sustain=0.6, release=0.3)

    sq_env = sq.data * env.data
    saw_env = saw.data * env.data

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].specgram(sq_env, Fs=SAMPLE_RATE, NFFT=2048, noverlap=1536, cmap='magma')
    axes[0].set_title("矩形波")
    axes[0].set_ylabel("周波数 (Hz)")
    axes[0].set_xlabel("時間 (秒)")
    axes[0].set_ylim(0, 5000)

    axes[1].specgram(saw_env, Fs=SAMPLE_RATE, NFFT=2048, noverlap=1536, cmap='magma')
    axes[1].set_title("ノコギリ波")
    axes[1].set_ylabel("周波数 (Hz)")
    axes[1].set_xlabel("時間 (秒)")
    axes[1].set_ylim(0, 5000)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig8_square_vs_saw_spectrogram.png", dpi=150)
    plt.close()
    print("fig8 done")


def fig9_chord_spectrum():
    """fig9: メジャーとマイナーのスペクトル比較"""
    dur = 1.5
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)

    c_major_notes = [60, 64, 67]
    c_minor_notes = [60, 63, 67]

    def make_chord(notes):
        freqs = [note_to_frequency(n) for n in notes]
        data = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
        return data, freqs

    major_data, major_freqs = make_chord(c_major_notes)
    minor_data, minor_freqs = make_chord(c_minor_notes)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for ax, (data, freqs, name) in zip(axes, [
        (major_data, major_freqs, "Cメジャー"),
        (minor_data, minor_freqs, "Cマイナー"),
    ]):
        windowed = data * np.hanning(len(data))
        mag = np.abs(np.fft.fft(windowed))
        fft_freqs = np.fft.fftfreq(len(windowed), 1 / SAMPLE_RATE)
        pos = fft_freqs >= 0
        rng = fft_freqs[pos] <= 1000

        ax.plot(fft_freqs[pos][rng], mag[pos][rng])
        ax.set_ylabel(f"{name}\n振幅")
        ax.grid(True, alpha=0.3)

        for f in freqs:
            ax.axvline(x=f, color='red', alpha=0.3, linestyle='--')

    axes[-1].set_xlabel("周波数 (Hz)")
    fig.suptitle("メジャーコードとマイナーコードのスペクトル", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig9_chord_spectrum.png", dpi=150)
    plt.close()
    print("fig9 done")


def fig10_reverb_spectrogram():
    """fig10: リバーブの効果をスペクトログラムで確認"""
    dur = 1.5
    saw = sawtooth_wave(440, dur)
    env = adsr(dur, attack=0.01, decay=0.3, sustain=0.5, release=0.3)
    dry_data = saw.data * env.data
    dry = AudioSignal(dry_data, SAMPLE_RATE)

    reverb = Reverb(room_size=0.8, damping=0.5, wet_level=0.5)
    wet = reverb.process(dry)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].specgram(dry.data, Fs=SAMPLE_RATE, NFFT=2048, noverlap=1536, cmap='magma')
    axes[0].set_title("ドライ")
    axes[0].set_xlabel("時間 (秒)")
    axes[0].set_ylabel("周波数 (Hz)")
    axes[0].set_ylim(0, 5000)

    axes[1].specgram(wet.data, Fs=SAMPLE_RATE, NFFT=2048, noverlap=1536, cmap='magma')
    axes[1].set_title("リバーブあり")
    axes[1].set_xlabel("時間 (秒)")
    axes[1].set_ylabel("周波数 (Hz)")
    axes[1].set_ylim(0, 5000)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig10_reverb_spectrogram.png", dpi=150)
    plt.close()
    print("fig10 done")


if __name__ == "__main__":
    fig1_simple_spectrum()
    fig2_four_waveforms_spectrum()
    fig3_window_problem()
    fig4_window_solved()
    fig5_window_functions()
    fig6_chirp_spectrogram()
    fig7_piano_analysis()
    fig8_square_vs_saw_spectrogram()
    fig9_chord_spectrum()
    fig10_reverb_spectrogram()
    print("全図の生成が完了しました")
