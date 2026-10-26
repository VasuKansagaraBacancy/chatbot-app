"""Repair text damaged by broken font encodings, before anything indexes it.

The GEP report embeds subsetted Type1 fonts whose ToUnicode maps are wrong for
a handful of glyphs. Every chapter was typeset with a different subset, so the
same ligature decodes differently depending on where it appears: "The" comes
out of the extractor as "7'_he", "—ó<he" or "Fﬂähe". Around 315 occurrences of
"The/This/These" are affected, which matters more than the count suggests —
those words open sentences, so a chunk can begin with garbage, and a reader
shown the citation sees nonsense.

Ligatures are a quieter version of the same problem. "ﬁscal" and "ﬂows" are
single code points, so a BM25 index built from them contains no token "fiscal",
and a user typing "fiscal" matches nothing by keyword. Dense retrieval degrades
too, since the embedder's tokenizer has never seen the ligature. Normalizing
them is the single highest-value text repair in the pipeline.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List

# Every observed mis-decoding of the "Th" ligature, longest first so that the
# shorter patterns cannot consume part of a longer one. Each is only applied
# where an "h" follows, which is the whole of the damaged ligature's context
# and keeps the substitution from touching ordinary prose.
_BROKEN_TH = re.compile(r"(?:Fﬂä|—ó<|7'_|7T_|TT_|T_)(?=h)")

# "f_inancial", "f_indings": the same fault in the "fi" ligature.
_BROKEN_FI = re.compile(r"f_(?=[aeiou])")

_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
    "Œ": "OE", "œ": "oe", "Æ": "AE", "æ": "ae",
}

# A word split across a line break: "govern-\nment" -> "government".
_HYPHEN_BREAK = re.compile(r"(\w)[-‐‑]\n(\w)")

_SPACES = re.compile(r"[ \t   ]+")
_BLANKS = re.compile(r"\n{3,}")


def repair(text: str) -> str:
    """Undo font-level damage. Safe to run on text that has none."""
    if not text:
        return ""
    text = _BROKEN_TH.sub("T", text)
    text = _BROKEN_FI.sub("f", text)
    for bad, good in _LIGATURES.items():
        if bad in text:
            text = text.replace(bad, good)
    return text


_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")

# How much rarer the halves must be than the whole before a pair is rejoined.
_JOIN_MARGIN = 4
_JOIN_MIN_FREQ = 3


def build_vocabulary(texts: List[str]) -> Counter:
    """Count the word forms the document actually uses."""
    vocab: Counter = Counter()
    for text in texts:
        for match in _WORD.finditer(text):
            vocab[match.group(0).lower()] += 1
    return vocab


def rejoin_split_words(text: str, vocab: Counter) -> str:
    """Undo spaces the extractor inserted inside a word.

    pypdf decides where the spaces go from the kerning adjustments in the
    PDF's text operators. In justified columns set in this report's serif
    face those adjustments grow large enough mid-word to look like word
    breaks, so "potential" arrives as "poten tial" and "these" as "th ese".
    The keyword index then holds neither word, and the embedding is computed
    from tokens the model has never seen.

    No spelling dictionary is available offline, so the document supplies its
    own: the same words occur correctly elsewhere, hundreds of times. A pair
    is rejoined only when the whole is well attested and both halves are much
    rarer than the whole, which leaves ordinary pairs like "in to" alone -
    "in" and "to" are each far commoner than "into".
    """
    if not vocab:
        return text

    def fix(line: str) -> str:
        parts = line.split(" ")
        out: List[str] = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts):
                left, right = parts[i], parts[i + 1]
                lead = _WORD.fullmatch(left)
                tail = _WORD.fullmatch(right)
                if lead and tail:
                    joined = (left + right).lower()
                    whole = vocab.get(joined, 0)
                    halves = min(vocab.get(left.lower(), 0),
                                 vocab.get(right.lower(), 0))
                    if whole >= _JOIN_MIN_FREQ and halves * _JOIN_MARGIN <= whole:
                        out.append(left + right)
                        i += 2
                        continue
            out.append(parts[i])
            i += 1
        return " ".join(out)

    return "\n".join(fix(line) for line in text.split("\n"))


def normalize(text: str) -> str:
    """Repair, de-hyphenate across line breaks, and tidy whitespace.

    Line breaks themselves are preserved: the chunker reads them as block
    boundaries, and flattening them here would destroy heading detection.
    """
    text = repair(text.replace("\r\n", "\n"))
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANKS.sub("\n\n", text)
    return text.strip()
