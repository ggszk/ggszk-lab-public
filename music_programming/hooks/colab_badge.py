"""MkDocs hook: Python 回の章に Open in Colab バッジを自動挿入する。

mkdocs.yml で hooks: [docs/hooks/colab_badge.py] と設定して使う。
"""

import re

# ノートブックが存在する Python 回の章ファイル名
PYTHON_CHAPTERS = {
    "ch01_sound_and_digital_audio",
    "ch02_waveforms_and_timbre",
    "ch04_harmonics_and_timbre",
    "ch06_scales_and_frequency",
    "ch08_envelope_adsr",
    "ch09_effects",
    "ch11_rhythm_and_score",
    "ch13_spectrum_analysis",
}

COLAB_BASE = (
    "https://colab.research.google.com/github/ggszk/"
    "ggszk-lab-public/blob/main/music_programming/notebooks"
)

BADGE_TEMPLATE = (
    '[![Open in Colab]'
    '(https://colab.research.google.com/assets/colab-badge.svg)]'
    '({url})'
    '{{ .colab-badge }}\n\n'
)


def on_page_markdown(markdown, page, config, files):
    """# 見出しの直後に Colab バッジを挿入する。"""
    src_path = page.file.src_path  # e.g. "chapters/ch01_sound_and_digital_audio.md"

    # chapters/ 配下の Python 回のみ対象
    m = re.search(r"chapters/(ch\d+_\w+)\.md$", src_path)
    if not m:
        return markdown
    chapter_name = m.group(1)
    if chapter_name not in PYTHON_CHAPTERS:
        return markdown

    url = f"{COLAB_BASE}/{chapter_name}.ipynb"
    badge = BADGE_TEMPLATE.format(url=url)

    # 最初の # 見出しの直後に挿入
    markdown = re.sub(
        r"(^# .+\n)",
        r"\1\n" + badge,
        markdown,
        count=1,
        flags=re.MULTILINE,
    )
    return markdown
