"""Chinese text normalization and tokenization (stdlib only)."""

from __future__ import annotations

import re

CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
ASCII_RE = re.compile(r"[A-Za-z0-9]+")


def normalize(text: str) -> str:
    """Keep only CJK letters and ASCII alphanumerics.

    Chinese book-title marks and punctuation (《》, 、, 第...条 separators,
    etc.) are removed so that law+article references can be compared across
    slightly different formats.
    """
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", text)


def tokens(text: str) -> list[str]:
    """Token stream for Chinese text.

    - ASCII words are kept as whole lowercase tokens;
    - CJK runs are split into character bigrams (plus unigrams for
      single-character runs), which works without a segmentation model.
    """
    t = normalize(text)
    out: list[str] = []
    for w in ASCII_RE.findall(t):
        out.append("w:" + w.lower())
    for run in CJK_RE.findall(t):
        if len(run) == 1:
            out.append("c:" + run)
        else:
            out.extend("c:" + run[i : i + 2] for i in range(len(run) - 1))
    return out


CN_NUM = {
    "0": "零", "1": "一", "2": "二", "3": "三", "4": "四", "5": "五",
    "6": "六", "7": "七", "8": "八", "9": "九",
}


def to_chinese_numeral(n: int) -> str:
    """1..99 -> Chinese numeral, e.g. 28 -> 二十八."""
    if n <= 0 or n >= 100:
        return str(n)
    if n < 10:
        return CN_NUM[str(n)]
    tens, ones = divmod(n, 10)
    s = ("" if tens == 1 else CN_NUM[str(tens)]) + "十"
    return s + (CN_NUM[str(ones)] if ones else "")
