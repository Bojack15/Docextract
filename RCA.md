# Root Cause Analysis: Production Inefficiencies in Document Ingestion

During early runs of the document ingestion pipeline, we encountered several performance issues, data leakage bottlenecks, and quality drops. This document outlines the findings, root causes, and resolutions implemented to stabilize the ingestion process.

---

## 1. Out-of-Memory (OOM) Errors on Multi-Page PDFs
**Context**: Processing large, scanned PDFs (50+ pages) frequently caused container crashes and memory spikes.
- **Root Cause**: The naive implementation loaded the entire PDF into memory using `pdf2image.convert_from_path()`. If a single page failed standard text extraction, the system rendered *every* page of the PDF into high-DPI images at once, leaking file handles and exhausting system RAM.
- **Impact**: Server failures on long documents; excessive memory footprint.
- **Resolution**: Implemented lazy, page-by-page rendering. The PDF converter now targets only the specific scanned page index (`first_page=i, last_page=i`) when OCR is triggered. Memory utilization scales $O(1)$ with page size instead of $O(N)$ with document length.

---

## 2. Low Throughput During OCR Processing
**Context**: Extracting text from mixed-media documents took up to 8 seconds per page.
- **Root Cause**: Inefficient OCR pipeline execution. The extraction module ran Tesseract OCR twice per image page: first via `pytesseract.image_to_data()` to calculate word confidence, and second via `pytesseract.image_to_string()` to extract page text.
- **Impact**: High CPU consumption, slow batch processing times, and double the necessary API latency.
- **Resolution**: Refactored the OCR routine into a single-pass implementation. We retrieve word-level confidence and reconstruct page geometry (blocks, lines, and spaces) directly from the `image_to_data` response, completely eliminating the duplicate `image_to_string` run. This cut OCR processing time by approximately 50%.

---

## 3. High Context Fragmentation in Vector Retrieval
**Context**: Semantic search queries returned disjointed sentences and poor RAG answers.
- **Root Cause**: Naive fixed-length text chunking split sentences in half and cut mid-paragraph. The semantic boundary of ideas was lost, and embedding queries matched noise.
- **Impact**: Inaccurate and fragmented retrieval results in the target application.
- **Resolution**: Developed a natural-boundary text splitter that prioritizes double-newline paragraph separation, recursively fallback-splitting on sentences only when paragraphs exceed target chunk size. Consecutive chunks retain configurable word-based overlaps to preserve semantic continuity.

---

## 4. Poor System Traceability and Lack of Structured State
**Context**: Inability to track where search matches originated or to reliably detect duplicate document uploads.
- **Root Cause**: Extracted text was treated as unstructured string blobs. There was no unique fingerprint representing document integrity and no systematic correlation to source pages or tables.
- **Impact**: Corrupted storage logs, data duplication in ChromaDB, and lost provenance.
- **Resolution**: Designed strict Pydantic models mapping data structures from `Document` to `Page` to `Chunk`. Integrated SHA-256 file hashing to prevent duplicate db ingestion, and structured page numbering mapping so every chunk explicitly references its source page range.
