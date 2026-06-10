import re
from dataclasses import dataclass
from models import Chunk


@dataclass
class ChunkConfig:
    size: int = 512
    overlap: int = 50
    min_size: int = 30


def chunk_text(
    text: str,
    document_id: str,
    config: ChunkConfig | None = None,
    page_breaks: dict[int, int] | None = None,
) -> list[Chunk]:
    if not text or not text.strip():
        return []

    config = config or ChunkConfig()
    paragraphs = _split_paragraphs(text)
    raw_chunks = _merge_paragraphs(paragraphs, config)

    chunks = []
    for i, chunk_str in enumerate(raw_chunks):
        start_page, end_page = _find_pages(chunk_str, text, page_breaks)
        words = len(chunk_str.split())

        chunks.append(Chunk(
            document_id=document_id,
            text=chunk_str,
            index=i,
            start_page=start_page,
            end_page=end_page,
            word_count=words,
        ))

    return chunks


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _merge_paragraphs(paragraphs: list[str], config: ChunkConfig) -> list[str]:
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        if para_words > config.size:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_words = 0

            chunks.extend(_split_large_paragraph(para, config))
            continue

        if current_words + para_words > config.size and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts, current_words = _take_overlap(current_parts, config.overlap)

        current_parts.append(para)
        current_words += para_words

    if current_parts:
        last = "\n\n".join(current_parts)
        if len(last.split()) >= config.min_size:
            chunks.append(last)
        elif chunks:
            chunks[-1] += "\n\n" + last

    return chunks or ["\n\n".join(paragraphs)]


def _split_large_paragraph(text: str, config: ChunkConfig) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        s_words = len(sentence.split())
        if current_words + s_words > config.size and current:
            chunks.append(" ".join(current))
            current = current[-1:]
            current_words = len(current[0].split()) if current else 0
        current.append(sentence)
        current_words += s_words

    if current:
        chunks.append(" ".join(current))

    return chunks


def _take_overlap(parts: list[str], max_words: int) -> tuple[list[str], int]:
    overlap: list[str] = []
    total = 0
    for part in reversed(parts):
        w = len(part.split())
        if total + w > max_words:
            break
        overlap.insert(0, part)
        total += w
    return overlap, total


def _find_pages(
    chunk_text: str,
    full_text: str,
    page_breaks: dict[int, int] | None,
) -> tuple[int, int]:
    if not page_breaks:
        return 1, 1

    pos = full_text.find(chunk_text)
    if pos == -1:
        return 1, 1

    end_pos = pos + len(chunk_text)
    offsets = sorted(page_breaks.keys())

    start_page = 1
    end_page = 1
    for offset in offsets:
        if offset <= pos:
            start_page = page_breaks[offset]
        if offset <= end_pos:
            end_page = page_breaks[offset]

    return start_page, end_page
