import logging
from pathlib import Path
# pyrefly: ignore [missing-import]
import chromadb
from chromadb.config import Settings
from models import Document

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, path: str = "./vector_db", collection: str = "documents"):
        self.path = str(Path(path).resolve())
        self.client = chromadb.PersistentClient(
            path=self.path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name=collection)
        logger.info("Connected to ChromaDB: '%s' (%d records)", collection, self.collection.count())

    def add(self, doc: Document) -> int:
        if not doc.chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for chunk in doc.chunks:
            ids.append(chunk.id)
            documents.append(chunk.text)
            metadatas.append({
                "document_id": doc.id,
                "filename": doc.filename,
                "chunk_index": chunk.index,
                "start_page": chunk.start_page,
                "end_page": chunk.end_page,
                "word_count": chunk.word_count,
                "method": doc.method.value,
                "total_pages": doc.total_pages,
            })

        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        logger.info("Indexed %d chunks for document '%s'", len(ids), doc.filename)
        return len(ids)

    def search(self, query: str, n: int = 5, where: dict | None = None) -> list[dict]:
        count = self.collection.count()
        if count == 0:
            return []

        kwargs = {"query_texts": [query], "n_results": min(n, count)}
        if where:
            kwargs["where"] = where

        raw = self.collection.query(**kwargs)
        
        results = []
        if raw and raw.get("documents") and len(raw["documents"]) > 0:
            for i, text in enumerate(raw["documents"][0]):
                results.append({
                    "text": text,
                    "metadata": raw["metadatas"][0][i],
                    "distance": raw["distances"][0][i],
                    "id": raw["ids"][0][i],
                })
        return results

    def delete(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})
        logger.info("Deleted chunks for document ID %s", document_id)

    def list_documents(self) -> list[dict]:
        data = self.collection.get(include=["metadatas"])
        if not data or not data.get("metadatas"):
            return []

        docs: dict[str, dict] = {}
        for meta in data["metadatas"]:
            did = meta.get("document_id")
            if not did:
                continue
            if did not in docs:
                docs[did] = {
                    "document_id": did,
                    "filename": meta.get("filename", "Unknown"),
                    "total_pages": meta.get("total_pages", 0),
                    "method": meta.get("method", "Unknown"),
                    "chunks": 0,
                }
            docs[did]["chunks"] += 1

        return list(docs.values())

    def stats(self) -> dict:
        docs = self.list_documents()
        return {
            "total_chunks": self.collection.count(),
            "total_documents": len(docs),
            "path": self.path,
            "documents": docs,
        }

    def reset(self) -> None:
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(name=name)
        logger.info("Reset collection '%s'", name)
