import hashlib
import uuid
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class ExtractionMethod(str, Enum):
    TEXT = "text"
    OCR = "ocr"
    HYBRID = "hybrid"
    OMR = "omr"


class Page(BaseModel):
    number: int
    text: str = ""
    method: ExtractionMethod = ExtractionMethod.TEXT
    confidence: float | None = None
    word_count: int = 0
    tables: list[list[list[str]]] = Field(default_factory=list)


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    text: str
    index: int
    start_page: int = 1
    end_page: int = 1
    word_count: int = 0

    def to_record(self) -> dict:
        return {
            "id": self.id,
            "document": self.text,
            "metadata": {
                "document_id": self.document_id,
                "chunk_index": self.index,
                "start_page": self.start_page,
                "end_page": self.end_page,
                "word_count": self.word_count,
            },
        }


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    filepath: str
    file_hash: str = ""
    file_size: int = 0
    total_pages: int = 0
    total_words: int = 0
    method: ExtractionMethod = ExtractionMethod.TEXT
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    pages: list[Page] = Field(default_factory=list)
    full_text: str = ""
    chunks: list[Chunk] = Field(default_factory=list)

    def summary(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "pages": self.total_pages,
            "words": self.total_words,
            "chunks": len(self.chunks),
            "method": self.method.value,
        }


def file_hash(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()
