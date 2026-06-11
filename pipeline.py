import json
import logging
import os
from pathlib import Path

from models import Document, ExtractionMethod, file_hash
from extractor import extract, SUPPORTED_IMAGES
from chunker import ChunkConfig, chunk_text

logger = logging.getLogger(__name__)


def process_file(
    filepath: str,
    chunk_config: ChunkConfig | None = None,
    ocr_lang: str = "eng",
    dpi: int = 300,
    is_omr: bool = False,
) -> Document:
    filepath = str(Path(filepath).resolve())
    pages = extract(filepath, ocr_lang=ocr_lang, dpi=dpi, is_omr=is_omr)

    methods = {p.method for p in pages}
    if len(methods) > 1:
        method = ExtractionMethod.HYBRID
    else:
        method = methods.pop() if methods else ExtractionMethod.TEXT

    full_text_parts = []
    page_breaks: dict[int, int] = {}
    offset = 0

    for page in pages:
        page_content_parts = []
        if page.text and page.text.strip():
            page_content_parts.append(page.text.strip())

        for table in page.tables:
            table_str = _format_table(table)
            if table_str and table_str.strip():
                page_content_parts.append(table_str.strip())

        if not page_content_parts:
            continue

        page_content = "\n\n".join(page_content_parts)
        page_breaks[offset] = page.number
        full_text_parts.append(page_content)
        offset += len(page_content) + 2

    full_text = "\n\n".join(full_text_parts)
    total_words = sum(p.word_count for p in pages)

    doc = Document(
        filename=Path(filepath).name,
        filepath=filepath,
        file_hash=file_hash(filepath),
        file_size=os.path.getsize(filepath),
        total_pages=len(pages),
        total_words=total_words,
        method=method,
        pages=pages,
        full_text=full_text,
    )

    doc.chunks = chunk_text(
        text=full_text,
        document_id=doc.id,
        config=chunk_config,
        page_breaks=page_breaks,
    )

    logger.info(
        "Processed '%s': %d pages, %d words, %d chunks (%s)",
        doc.filename, len(pages), total_words, len(doc.chunks), method.value
    )

    return doc


def process_directory(
    dir_path: str,
    chunk_config: ChunkConfig | None = None,
    ocr_lang: str = "eng",
    dpi: int = 300,
    recursive: bool = True,
    is_omr: bool = False,
) -> list[Document]:
    extensions = {".pdf"} | SUPPORTED_IMAGES
    dir_path = Path(dir_path)

    if not dir_path.is_dir():
        raise NotADirectoryError(str(dir_path))

    pattern = "**/*" if recursive else "*"
    files = sorted(
        f for f in dir_path.glob(pattern)
        if f.is_file() and f.suffix.lower() in extensions
    )

    logger.info("Found %d supported files in %s", len(files), dir_path)

    results = []
    for f in files:
        try:
            doc = process_file(str(f), chunk_config, ocr_lang, dpi, is_omr=is_omr)
            results.append(doc)
        except Exception as e:
            logger.error("Skipped '%s' due to error: %s", f.name, e)

    return results


def export_json(doc: Document, output_path: str) -> str:
    output_path = str(Path(output_path).resolve())
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    data = doc.model_dump()
    for chunk in data.get("chunks", []):
        chunk.pop("embedding", None)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Exported JSON structure to %s", output_path)
    return output_path


def _format_table(table: list[list[str]]) -> str:
    if not table:
        return ""
    return "\n".join(" | ".join(row) for row in table)
