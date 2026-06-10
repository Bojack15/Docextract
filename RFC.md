# RFC: DocExtract Ingestion & Vector Storage Pipeline

## 1. Objective & Scope
The objective is to design and implement a robust Python pipeline capable of extracting structured text and tabular data from both native and scanned documents (PDFs and image formats). The output should be validated, chunked along natural semantic boundaries, and ingested into a local vector database (ChromaDB) to support zero-dependency semantic search.

---

## 2. Architecture & Data Flow

```
                     +---------------------------------------+
                     |         Input: PDF / Image            |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |             extractor.py              |
                     | - Checks page structures natively     |
                     | - OCR fallback via single-pass data   |
                     +---------------------------------------+
                                         |
                                         | [Output: Page Models]
                                         v
                     +---------------------------------------+
                     |              chunker.py               |
                     | - Paragraph & sentence segmentation   |
                     | - Respects overlap constraint         |
                     +---------------------------------------+
                                         |
                                         | [Output: Chunk Models]
                                         v
                     +---------------------------------------+
                     |           vector_store.py             |
                     | - SHA-256 duplicate deduplication     |
                     | - ChromaDB indexing (MiniLM embeddings) |
                     +---------------------------------------+
```

### Module Responsibilities:
- `models.py`: Structural data boundaries using Pydantic validation (Document, Page, Chunk).
- `extractor.py`: Hybrid reader utilizing `pdfplumber` for text/tables and lazy `pytesseract` for scanned material.
- `chunker.py`: Boundary-preserving text chunking engine.
- `pipeline.py`: Process coordinator mapping system files/directories to vector store ingestion.
- `vector_store.py`: ChromaDB integration layer managing search and persistence.

---

## 3. Design Decisions & Trade-offs

### 3.1 OCR Laziness and Page Boundary Isolation
* **Approach**: We isolate OCR triggers to individual pages using range-restricted PDF conversion: `pdf2image.convert_from_path(..., first_page=i, last_page=i)`.
* **Trade-off**: While opening the file handle multiple times adds negligible OS overhead, it avoids loading massive multi-megabyte image arrays into memory simultaneously. This allows processing large PDFs on memory-constrained systems.

### 3.2 Single-Pass OCR Text & Metadata Extraction
* **Approach**: Tesseract OCR is invoked exactly once per scanned page using `image_to_data`. The plain text page layout is programmatically reconstructed from the returned dictionary.
* **Trade-off**: This prevents running OCR twice (once for plain text, once for layout confidence), cutting compute time in half at the cost of maintaining a custom dictionary parsing routine.

### 3.3 Target Chunk Sizing by Word Count
* **Approach**: Semantic chunk boundaries are evaluated via word counts rather than strict tokenizer token counts.
* **Trade-off**: Word counting is model-agnostic and fast. If the embedding model changes, the chunking code remains valid without importing heavy tokenizer libraries. However, it can occasionally lead to minor token count discrepancies with large vocabularies.

### 3.4 Local SQLite-based Embedding Database
* **Approach**: ChromaDB is used with persistent local storage and standard all-MiniLM-L6-v2 embeddings.
* **Trade-off**: This ensures zero network latency and no external cloud dependency, which is excellent for security and self-contained testing. However, it is limited by single-host storage scale and CPU-bound search performance.

---

## 4. API Specification

```python
from pipeline import process_file
from vector_store import VectorStore
from chunker import ChunkConfig

# 1. Process document
config = ChunkConfig(size=256, overlap=30)
doc = process_file("Q2_report.pdf", chunk_config=config)

# 2. Persist to vector database
store = VectorStore(path="./vector_db")
store.add(doc)

# 3. Retrieve relevant context
results = store.search("quarterly financial updates", n=3)
for result in results:
    print(f"Relevance: {1 - result['distance']:.2%}")
    print(f"Content: {result['text'][:150]}...")
```
