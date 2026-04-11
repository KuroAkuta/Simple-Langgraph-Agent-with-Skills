"""
Document Indexer service for chunking and vectorizing documents.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from http import HTTPStatus

import dashscope
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

from knowledge.chunk_tracker import ChunkTracker
from knowledge.models import IndexingStrategy, DocumentInfo, DocumentStatus
from services.knowledge_manager import KnowledgeManager
from config.settings import settings


class DashScopeEmbeddings:
    """DashScope embeddings wrapper."""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        # DashScope limits batch size to 10
        self.batch_size = 10

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents.

        DashScope API limits batch size to 10, so we process in batches.
        """
        import dashscope
        from http import HTTPStatus

        all_embeddings = []

        # Process in batches to avoid exceeding the limit
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            resp = dashscope.TextEmbedding.call(
                model=self.model,
                input=batch_texts,
                api_key=self.api_key
            )

            if resp.status_code == HTTPStatus.OK:
                batch_embeddings = [item["embedding"] for item in resp.output["embeddings"]]
                all_embeddings.extend(batch_embeddings)
            else:
                raise Exception(f"DashScope error: {resp.code} - {resp.message}")

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query."""
        return self.embed_documents([text])[0]


class DocumentIndexer:
    """
    Handles document chunking, embedding, and vector storage.

    Supports both incremental and full indexing strategies.
    """

    def __init__(
        self,
        persist_dir: str = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        """
        Initialize the document indexer.

        Args:
            persist_dir: Directory for persisting vector store
            embedding_model: Name of the embedding model to use
        """
        if persist_dir is None:
            self.persist_dir = Path(__file__).parent.parent / "storage" / "chroma"
        else:
            self.persist_dir = Path(persist_dir)

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize DashScope embeddings
        self.embeddings = DashScopeEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.EMBEDDING_API_KEY or settings.CUSTOM_API_KEY,
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "。", "!", "!", "......", ".", " ", ""],
        )

    def _get_vectorstore(self, kb_id: str) -> Chroma:
        """
        Get or create a vector store for a knowledge base.

        Args:
            kb_id: Knowledge base identifier

        Returns:
            Chroma vector store instance
        """
        collection_name = f"kb_{kb_id}_chunks"

        return Chroma(
            embedding_function=self.embeddings,
            collection_name=collection_name,
            persist_directory=str(self.persist_dir),
        )

    def _read_document_content(
        self, file_path: Path, filename: str
    ) -> str:
        """
        Read content from a document file.

        Supports multiple formats:
        - Plain text: .txt, .md, .csv, .json, .py, .js, etc.
        - PDF: .pdf (using markitdown)
        - Word: .doc, .docx (using markitdown)
        - PowerPoint: .ppt, .pptx (using markitdown)
        - Excel: .xls, .xlsx (using markitdown)

        Args:
            file_path: Path to the document file
            filename: Original filename (for format detection)

        Returns:
            Document content as text
        """
        suffix = Path(filename).suffix.lower()

        # Supported document formats for markitdown conversion
        document_formats = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}

        # Plain text formats - read directly
        text_formats = {
            ".txt", ".md", ".csv", ".json", ".py", ".js", ".ts",
            ".jsx", ".tsx", ".yaml", ".yml", ".toml", ".ini",
            ".cfg", ".conf", ".log", ".xml", ".html", ".htm",
        }

        if suffix in text_formats:
            # Read plain text files with encoding fallback
            encodings = ["utf-8", "gbk", "latin-1"]
            for encoding in encodings:
                try:
                    content = file_path.read_text(encoding=encoding)
                    return content
                except UnicodeDecodeError:
                    continue

            # If all text encodings fail, try reading as binary and decode
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                return content
            except Exception as e:
                raise ValueError(f"Cannot read file {filename}: {e}")

        elif suffix in document_formats:
            # Use markitdown for document files
            try:
                from markitdown import MarkItDown
                md = MarkItDown()
                result = md.convert(str(file_path))
                return result.text_content
            except ImportError:
                raise ValueError(
                    f"markitdown library not installed. "
                    f"Please install it to support {suffix} files: pip install markitdown"
                )
            except Exception as e:
                raise ValueError(f"Cannot parse {suffix} file {filename}: {e}")

        else:
            # Unknown format - try to read as plain text
            try:
                content = file_path.read_text(encoding="utf-8")
                return content
            except UnicodeDecodeError:
                raise ValueError(
                    f"Unsupported file format: {suffix}. "
                    f"Supported formats: {', '.join(sorted(document_formats | text_formats))}"
                )

    def index_documents(
        self,
        kb_id: str,
        documents: List[DocumentInfo],
        knowledge_manager: KnowledgeManager,
        strategy: IndexingStrategy = IndexingStrategy.INCREMENTAL,
    ) -> Tuple[int, int]:
        """
        Index a list of documents.

        Args:
            kb_id: Knowledge base identifier
            documents: List of DocumentInfo objects to index
            knowledge_manager: KnowledgeManager instance
            strategy: Indexing strategy (incremental or full)

        Returns:
            Tuple of (total_chunks, new_chunks)
        """
        if strategy == IndexingStrategy.FULL:
            # Full mode: delete existing collection and clear tracker
            vectorstore = self._get_vectorstore(kb_id)
            try:
                vectorstore.delete_collection()
            except Exception:
                pass  # Collection might not exist

            chunk_tracker = ChunkTracker(kb_id, str(self.persist_dir.parent / "knowledge"))
            chunk_tracker.clear()

        chunk_tracker = ChunkTracker(kb_id, str(self.persist_dir.parent / "knowledge"))
        vectorstore = self._get_vectorstore(kb_id)

        total_new_chunks = 0
        total_chunks = 0

        for doc in documents:
            # Skip only successfully indexed documents in incremental mode
            # FAILED documents should be re-indexed
            if doc.status == DocumentStatus.INDEXED and strategy == IndexingStrategy.INCREMENTAL:
                # Skip already indexed documents in incremental mode
                doc_path = knowledge_manager.get_document_file_path(kb_id, doc.id)
                if doc_path:
                    try:
                        content = self._read_document_content(doc_path, doc.filename)
                        temp_chunks = self.text_splitter.split_text(content)
                        total_chunks += len(temp_chunks)
                    except Exception:
                        pass
                continue

            # Update status to indexing (this will clear any previous FAILED status)
            knowledge_manager.update_document_status(kb_id, doc.id, DocumentStatus.INDEXING)

            try:
                # Get document content
                doc_path = knowledge_manager.get_document_file_path(kb_id, doc.id)
                if not doc_path:
                    raise FileNotFoundError(f"Document file not found: {doc.id}")

                content = self._read_document_content(doc_path, doc.filename)

                # Split into chunks
                chunks = self.text_splitter.split_text(content)

                # Build chunks with proper metadata
                chunk_docs = []
                chunk_hashes = []

                for i, chunk_text in enumerate(chunks):
                    chunk_hash = chunk_tracker.compute_chunk_hash(chunk_text, doc.id, i)
                    chunk_hashes.append(chunk_hash)

                    chunk_docs.append(
                        Document(
                            page_content=chunk_text,
                            metadata={
                                "doc_id": doc.id,
                                "doc_name": doc.filename,
                                "kb_id": kb_id,
                                "chunk_idx": i,
                            },
                        )
                    )

                # Filter already indexed chunks (incremental mode)
                new_chunk_docs = []
                new_chunk_hashes = []

                for i, chunk_doc in enumerate(chunk_docs):
                    if not chunk_tracker.is_chunk_indexed(chunk_hashes[i]):
                        new_chunk_docs.append(chunk_doc)
                        new_chunk_hashes.append(chunk_hashes[i])

                if new_chunk_docs:
                    # Add to vector store
                    vectorstore.add_documents(new_chunk_docs)
                    chunk_tracker.mark_chunks_indexed(new_chunk_hashes)

                    total_new_chunks += len(new_chunk_docs)

                # Update document status
                knowledge_manager.update_document_status(
                    kb_id,
                    doc.id,
                    DocumentStatus.INDEXED,
                    chunk_count=len(chunks),
                )

                total_chunks += len(chunks)

            except Exception as e:
                # Update document status to failed
                knowledge_manager.update_document_status(
                    kb_id,
                    doc.id,
                    DocumentStatus.FAILED,
                    error_message=str(e),
                )
                raise

        # Update knowledge base total chunks
        if kb := knowledge_manager.get_knowledge_base(kb_id):
            kb.total_chunks = total_chunks
            knowledge_manager._save_metadata()

        return (total_chunks, total_new_chunks)

    def index_single_document(
        self,
        kb_id: str,
        doc: DocumentInfo,
        knowledge_manager: KnowledgeManager,
        strategy: IndexingStrategy = IndexingStrategy.INCREMENTAL,
    ) -> int:
        """
        Index a single document.

        Args:
            kb_id: Knowledge base identifier
            doc: DocumentInfo object
            knowledge_manager: KnowledgeManager instance
            strategy: Indexing strategy

        Returns:
            Number of chunks created
        """
        total, new = self.index_documents(kb_id, [doc], knowledge_manager, strategy)
        return new

    def search(
        self,
        kb_id: str,
        query: str,
        k: int = 3,
        use_rerank: bool = False,
        top_n: int = 3,
    ) -> List[Document]:
        """
        Search for relevant chunks in a knowledge base.

        Args:
            kb_id: Knowledge base identifier
            query: Search query
            k: Number of results to retrieve initially
            use_rerank: Whether to use reranker for better results
            top_n: Number of final results after reranking

        Returns:
            List of relevant Document objects
        """
        vectorstore = self._get_vectorstore(kb_id)

        # Get more results for reranking
        retrieve_k = k * 3 if use_rerank else k
        results = vectorstore.similarity_search(query, k=retrieve_k)

        if use_rerank and results and settings.RERANKER_MODEL:
            try:
                from services.reranker import DashScopeRerank
                reranker = DashScopeRerank()

                texts = [doc.page_content for doc in results]
                reranked = reranker.rerank(query, texts, top_n=top_n)

                # Reorder results based on reranking
                reranked_results = []
                for item in reranked:
                    idx = item["index"]
                    if idx < len(results):
                        doc = results[idx]
                        doc.metadata["rerank_score"] = item["score"]
                        reranked_results.append(doc)

                return reranked_results[:top_n]

            except Exception as e:
                print(f"Rerank failed, falling back to similarity search: {e}")

        return results[:k]

    def search_multi(
        self,
        kb_ids: List[str],
        query: str,
        k_per_kb: int = 2,
        use_rerank: bool = False,
        top_n: int = 5,
    ) -> List[Document]:
        """
        Search across multiple knowledge bases.

        Args:
            kb_ids: List of knowledge base IDs to search
            query: Search query
            k_per_kb: Number of results per knowledge base
            use_rerank: Whether to use reranker
            top_n: Final number of results after reranking

        Returns:
            List of relevant Document objects
        """
        if use_rerank and settings.RERANKER_MODEL:
            # Collect all texts from all KBs for reranking
            all_texts = []
            all_docs = []

            for kb_id in kb_ids:
                try:
                    results = self.search(kb_id, query, k=k_per_kb * 2)
                    for doc in results:
                        all_texts.append(doc.page_content)
                        all_docs.append(doc)
                except Exception:
                    continue

            if all_texts:
                try:
                    from services.reranker import DashScopeRerank
                    reranker = DashScopeRerank()
                    reranked = reranker.rerank(query, all_texts, top_n=top_n)

                    reranked_docs = []
                    for item in reranked:
                        idx = item["index"]
                        if idx < len(all_docs):
                            doc = all_docs[idx]
                            doc.metadata["rerank_score"] = item["score"]
                            reranked_docs.append(doc)

                    return reranked_docs
                except Exception as e:
                    print(f"Rerank failed, falling back: {e}")

        # Default: just combine results from all KBs
        all_results = []
        for kb_id in kb_ids:
            try:
                results = self.search(kb_id, query, k=k_per_kb)
                all_results.extend(results)
            except Exception:
                continue

        return all_results[:top_n] if use_rerank else all_results

    def get_context_string(
        self,
        kb_ids: List[str],
        query: str,
        k_per_kb: int = 2,
        use_rerank: bool = True,
        top_n: int = 5,
    ) -> str:
        """
        Search and return context as a formatted string.

        Args:
            kb_ids: List of knowledge base IDs to search
            query: Search query
            k_per_kb: Number of results per knowledge base
            use_rerank: Whether to use reranker
            top_n: Max number of results

        Returns:
            Formatted context string
        """
        results = self.search_multi(kb_ids, query, k_per_kb, use_rerank, top_n)

        if not results:
            return ""

        context_parts = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("doc_name", "Unknown")
            score = doc.metadata.get("rerank_score")
            score_text = f" (score: {score:.2f})" if score else ""
            context_parts.append(f"[Source: {source}{score_text}]\n{doc.page_content}")

        return "\n\n---\n\n".join(context_parts)

    def delete_document_chunks(
        self,
        kb_id: str,
        doc_id: str,
        chunk_tracker: Optional[ChunkTracker] = None,
    ) -> int:
        """
        Delete all chunks belonging to a document.

        Args:
            kb_id: Knowledge base identifier
            doc_id: Document identifier
            chunk_tracker: Optional ChunkTracker instance

        Returns:
            Number of chunks deleted from ChromaDB
        """
        if chunk_tracker is None:
            chunk_tracker = ChunkTracker(kb_id, str(self.persist_dir.parent / "knowledge"))

        # Remove from tracker first
        chunks_removed = chunk_tracker.remove_doc_chunks(doc_id)

        # Delete from ChromaDB using metadata filter
        try:
            vectorstore = self._get_vectorstore(kb_id)
            if vectorstore:
                # Chroma supports delete by where filter
                deleted = vectorstore.delete(where={"doc_id": doc_id})
                # deleted is a dict with 'ids' key containing deleted IDs
                return len(deleted.get("ids", [])) if deleted else 0
        except Exception as e:
            print(f"Warning: Failed to delete vectors from ChromaDB: {e}")

        return chunks_removed

    def delete_knowledge_base_vectors(self, kb_id: str) -> int:
        """
        Delete all vectors belonging to a knowledge base.

        This deletes the entire ChromaDB collection for the KB.

        Args:
            kb_id: Knowledge base identifier

        Returns:
            Number of vectors deleted
        """
        try:
            vectorstore = self._get_vectorstore(kb_id)
            if vectorstore:
                # Get count before deletion
                count = len(vectorstore.get()["ids"]) if vectorstore.get()["ids"] else 0
                # Delete entire collection
                vectorstore._client.delete_collection(name=vectorstore._collection.name)
                return count
        except Exception as e:
            print(f"Warning: Failed to delete knowledge base vectors: {e}")
        return 0

    def get_stats(self, kb_id: str) -> Dict:
        """
        Get statistics for a knowledge base.

        Args:
            kb_id: Knowledge base identifier

        Returns:
            Dictionary with statistics
        """
        chunk_tracker = ChunkTracker(kb_id, str(self.persist_dir.parent / "knowledge"))
        tracker_stats = chunk_tracker.get_stats()

        try:
            vectorstore = self._get_vectorstore(kb_id)
            # Note: Chroma count may require collection access
            vector_count = len(vectorstore.get()["ids"]) if vectorstore.get()["ids"] else 0
        except Exception:
            vector_count = 0

        return {
            "kb_id": kb_id,
            "indexed_chunks": tracker_stats["total_chunks"],
            "vector_count": vector_count,
        }
