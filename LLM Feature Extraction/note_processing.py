"""
Text cleaning and chunking for clinical notes.

A full discharge note is too long to send to the model in one go, so this
module prepares it. It first cleans up the raw MIMIC formatting (de-identifi-
cation placeholders, wrapped lines, extra whitespace), splits the note into
sentences, and then builds short overlapping chunks. Two chunking strategies
are provided: keyword-based (keep only sentences near a feature keyword plus
some context) and full-note (split the whole note into a few equal parts).
"""

import re
from typing import List


def clean_raw_text(text: str) -> str:
    """Clean raw MIMIC note text into something easier to sentence-split.

    Removes the de-identification placeholders (like [**...**] and ___),
    collapses repeated spaces and blank lines, and normalizes line endings.
    Returns an empty string if the input is not a string.
    """
    if not isinstance(text, str):
        return ""

    text = text.replace("\r", "\n")
    text = text.replace("___", " ")
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def is_lab_like_line(line: str) -> bool:
    """Return True if a line looks like a lab result line.

    Used to keep lab lines (lactate, glucose, blood gas values, etc.) on their
    own instead of merging them into surrounding prose, since they are often
    wrapped or written as short fragments.
    """
    return bool(re.search(
        r"\b(lactate|glucose|ph|hco3|anion gap|base xs|wbc|hgb|hct|blood)\b",
        line,
        flags=re.I
    ))


def normalize_wrapped_lines(text: str) -> str:
    """Join lines that were hard-wrapped mid-sentence back together.

    Clinical notes often break a single sentence across several lines. This
    merges those lines into one until it hits sentence-ending punctuation or a
    blank line, but keeps lab-like lines separate so their values are not glued
    onto neighbouring text.
    """
    if not text:
        return ""

    lines = [line.strip() for line in text.split("\n")]
    merged = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            merged.append(current.strip())
        current = ""

    for line in lines:
        if not line:
            flush()
            continue

        if is_lab_like_line(line):
            flush()
            merged.append(line)
            continue

        if not current:
            current = line
        elif re.search(r"[.!?:]\s*$", current):
            flush()
            current = line
        else:
            current += " " + line

    flush()
    return "\n".join(merged)


def split_into_sentences(text: str) -> List[str]:
    """Split cleaned note text into a list of sentences.

    First normalizes wrapped lines, then splits each line on sentence-ending
    punctuation. Lab-like lines are kept whole as single "sentences".
    """
    if not text:
        return []

    text = normalize_wrapped_lines(text)
    sentences = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        if is_lab_like_line(line):
            sentences.append(line)
            continue

        parts = re.split(r'(?<=[\.\?\!])\s+', line)
        for p in parts:
            p = p.strip()
            if p:
                sentences.append(p)

    return sentences


def select_keyword_sentences(
    text: str,
    keywords: List[str],
    window: int = 0,
) -> List[str]:
    """Keep only sentences that mention a keyword, plus nearby context.

    Cleans and sentence-splits the note, then keeps every sentence that contains
    one of the keywords together with `window` sentences on each side. This is
    how the pipeline zooms in on the parts of a note that are relevant to a
    feature instead of feeding the whole note to the model.
    """
    text = clean_raw_text(text)
    sentences = split_into_sentences(text)

    if not sentences:
        return []

    keywords_lower = [k.lower() for k in keywords]
    keep_idxs = set()

    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        if any(k in sent_lower for k in keywords_lower):
            for j in range(max(0, i - window), min(len(sentences), i + window + 1)):
                keep_idxs.add(j)

    return [sentences[i] for i in sorted(keep_idxs)]


def make_chunks(sentences: List[str], chunk_size: int = 5, overlap: int = 1) -> List[str]:
    """Group sentences into overlapping chunks of `chunk_size` sentences.

    Consecutive chunks share `overlap` sentences so a piece of evidence that
    sits on a chunk boundary is not lost. Each chunk is returned as a single
    joined string.
    """
    if not sentences:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)

    for start in range(0, len(sentences), step):
        chunk = sentences[start:start + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
        if start + chunk_size >= len(sentences):
            break

    return chunks


def prepare_chunks(
    note_text: str,
    keywords: List[str],
    window: int = 1,
    chunk_size: int = 5,
    overlap: int = 1,
) -> List[str]:
    """Run the full keyword-based chunking for one note.

    Selects the keyword sentences (with context) and then packs them into
    overlapping chunks. This is the main entry point used by the keyword
    chunking mode.
    """
    selected_sentences = select_keyword_sentences(
        text=note_text,
        keywords=keywords,
        window=window,
    )

    return make_chunks(
        sentences=selected_sentences,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def prepare_full_note_sentence_chunks(
    note_text: str,
    n_chunks: int = 4,
) -> List[str]:
    """
    Split the full cleaned note into roughly equal-sized chunks based on sentence count.
    This bypasses keyword sentence selection entirely.
    """
    text = clean_raw_text(note_text)
    sentences = split_into_sentences(text)

    if not sentences:
        return []

    total = len(sentences)
    chunk_len = max(1, (total + n_chunks - 1) // n_chunks)  # ceiling division

    chunks = []
    for start in range(0, total, chunk_len):
        chunk = " ".join(sentences[start:start + chunk_len]).strip()
        if chunk:
            chunks.append(chunk)

    return chunks