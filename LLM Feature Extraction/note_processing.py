import re
from typing import List


def clean_raw_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.replace("\r", "\n")
    text = text.replace("___", " ")
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def is_lab_like_line(line: str) -> bool:
    return bool(re.search(
        r"\b(lactate|glucose|ph|hco3|anion gap|base xs|wbc|hgb|hct|blood)\b",
        line,
        flags=re.I
    ))


def normalize_wrapped_lines(text: str) -> str:
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