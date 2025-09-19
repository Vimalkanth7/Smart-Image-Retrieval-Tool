# src/clean.py
import re
from typing import List
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are available (run once: nltk.download("stopwords"))
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))

# Keep some words even if they appear in default stopwords
ALLOWLIST = {"up", "down", "near", "over", "under", "top", "bottom"}

_punct_re = re.compile(r"[^\w\s'-]+", re.UNICODE)
_multi_space_re = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation, normalize whitespace"""
    s = s.lower().strip()
    s = _punct_re.sub(" ", s)
    s = _multi_space_re.sub(" ", s)
    return s.strip()


def collapse_consecutive_duplicates(tokens: List[str], max_repeat: int = 1) -> List[str]:
    """Limit the same token repeated consecutively (che che che -> che)"""
    out = []
    last = None
    run = 0
    for t in tokens:
        if t == last:
            run += 1
        else:
            run = 1
            last = t
        if run <= max_repeat:
            out.append(t)
    return out


def remove_noise_tokens(tokens: List[str]) -> List[str]:
    """Drop stopwords, single-char noise, repeated nonsense tokens."""
    clean = []
    for t in tokens:
        if not t:
            continue

        # Drop stopwords unless in allowlist
        if t in STOPWORDS and t not in ALLOWLIST:
            continue

        # keep words that contain letters (allow hyphen/apostrophe)
        letters_only = re.sub(r"[^a-z'-]", "", t)
        if not letters_only or len(letters_only) == 0:
            continue

        # remove tokens that are just repeated single chars (e.g., 'aaaaa')
        if re.fullmatch(r"(.)\1{2,}", letters_only.replace("-", "").replace("'", "")):
            continue

        # single-letter tokens (except 'i' or 'a' when meaningful)
        if len(letters_only) == 1 and letters_only not in {"i", "a"}:
            continue

        clean.append(letters_only)
    return clean


def dedup_preserve_order(tokens: List[str], limit: int = 60) -> List[str]:
    """Remove duplicates but preserve order; cap length"""
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            out.append(t)
            seen.add(t)
        if len(out) >= limit:
            break
    return out


def clean_caption_and_keywords(raw_caption: str, keyword_top_k: int = 10) -> tuple[str, List[str]]:
    """
    1) normalize -> tokenize
    2) collapse immediate duplicates (che che che -> che)
    3) remove noisy tokens
    4) dedup keep order
    Returns (clean_caption, keywords)
    """
    base = normalize_text(raw_caption)
    toks = base.split()
    toks = collapse_consecutive_duplicates(toks, max_repeat=1)
    toks = remove_noise_tokens(toks)
    toks = dedup_preserve_order(toks, limit=60)

    if not toks:
        return ("photo", [])

    clean_caption = " ".join(toks)
    kws = toks[:keyword_top_k]
    return clean_caption, kws
