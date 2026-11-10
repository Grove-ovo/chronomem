"""Normalization utilities for the ChronoMem hybrid retriever.

Implements the PRD §3.4 text pipeline:

* Chinese -> **bigram** tokenization (every overlapping 2-char window),
  which turns FTS5's whitespace tokenizer into an effective CJK n-gram
  matcher without any custom tokenizer extension;
* Latin -> lowercase + light suffix stemming (``ing`` / ``ed`` / ``es`` /
  plural ``s`` with consonant-doubling repair), applied identically at
  index time and query time.
"""

from __future__ import annotations

import re
from typing import List

__all__ = ["normalize_tokens", "light_stem", "fts_query", "bigrams", "CJK_RE"]

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+|[a-z0-9]+")


def light_stem(word: str) -> str:
    """Conservative English stemmer; deterministic and applied on both sides."""
    w = word
    if len(w) > 4 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 3 and w.endswith("ed"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("es"):
        w = w[:-2]
    elif len(w) > 2 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    # repair consonant doubling: running -> runn -> run
    # ('s' excluded: legitimate ss-endings like process/address must survive)
    if len(w) > 2 and w[-1] == w[-2] and w[-1] not in "aeious":
        w = w[:-1]
    return w


def bigrams(run: str) -> List[str]:
    """Overlapping character bigrams of a CJK run; single char kept as-is."""
    if len(run) == 1:
        return [run]
    return [run[i : i + 2] for i in range(len(run) - 1)]


def normalize_tokens(text: str) -> List[str]:
    """Text -> normalized token list (CJK bigrams + stemmed latin words)."""
    lowered = text.lower()
    tokens: List[str] = []
    for run in _RUN_RE.findall(lowered):
        if CJK_RE.match(run):
            tokens.extend(bigrams(run))
        else:
            tokens.append(light_stem(run))
    return tokens


def tokens_to_text(tokens: List[str]) -> str:
    """Token list -> space-joined text suitable for FTS5 unicode61 storage."""
    return " ".join(tokens)


def fts_query(tokens: List[str]) -> str:
    """Token list -> FTS5 OR-query with each token double-quoted."""
    return " OR ".join(f'"{t}"' for t in tokens if t)
