"""
Document Indexer service for chunking and vectorizing documents.
"""
import os
import json
import pickle
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

        # BM25 index storage directory
        self.bm25_dir = Path(__file__).parent.parent / "storage" / "bm25"
        self.bm25_dir.mkdir(parents=True, exist_ok=True)

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

    def _get_bm25_index_path(self, kb_id: str) -> Path:
        """Get the file path for BM25 index of a knowledge base."""
        return self.bm25_dir / f"kb_{kb_id}_bm25.pkl"

    def _load_bm25_index(self, kb_id: str):
        """
        Load BM25 index for a knowledge base.

        Returns:
            BM25 index object or None if not exists
        """
        index_path = self._get_bm25_index_path(kb_id)
        if index_path.exists():
            try:
                with open(index_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load BM25 index for {kb_id}: {e}")
        return None

    def _save_bm25_index(self, kb_id: str, bm25_index):
        """Save BM25 index for a knowledge base."""
        index_path = self._get_bm25_index_path(kb_id)
        with open(index_path, "wb") as f:
            pickle.dump(bm25_index, f)

    def _load_bm25_corpus(self, kb_id: str) -> List[dict]:
        """
        Load BM25 corpus (list of documents with metadata) for a knowledge base.
        Filters out any invalid entries that lack proper metadata.

        Returns:
            List of dicts with 'text' and 'metadata' fields, or empty list if not exists
        """
        corpus_path = self.bm25_dir / f"kb_{kb_id}_corpus.json"
        if corpus_path.exists():
            try:
                with open(corpus_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    # Filter out invalid entries (must have 'text' and 'metadata')
                    return [
                        item for item in raw_data
                        if isinstance(item, dict) and "text" in item and "metadata" in item
                    ]
            except Exception as e:
                print(f"Warning: Failed to load BM25 corpus for {kb_id}: {e}")
        return []

    def _save_bm25_corpus(self, kb_id: str, corpus: List[dict]):
        """Save BM25 corpus for a knowledge base."""
        corpus_path = self.bm25_dir / f"kb_{kb_id}_corpus.json"
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.

        Uses jieba for Chinese tokenization.
        """
        try:
            import jieba
            # Use cut for precise mode
            return list(jieba.cut(text))
        except ImportError:
            # Fallback: simple split by whitespace and punctuation
            import re
            return re.findall(r'\w+', text)

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

            # Clear BM25 corpus and index
            bm25_corpus = []
            bm25_index = None
        else:
            # Incremental mode: load existing BM25 corpus
            bm25_corpus = self._load_bm25_corpus(kb_id)
            bm25_index = self._load_bm25_index(kb_id)

        chunk_tracker = ChunkTracker(kb_id, str(self.persist_dir.parent / "knowledge"))
        vectorstore = self._get_vectorstore(kb_id)

        total_new_chunks = 0
        total_chunks = 0
        new_bm25_docs = []  # Track new chunks for BM25

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

                        # Check if BM25 corpus is missing this document's chunks
                        # If so, add them to ensure consistency
                        existing_bm25_doc_ids = {
                            item["metadata"]["doc_id"]
                            for item in bm25_corpus
                            if "metadata" in item and "doc_id" in item["metadata"]
                        }
                        if doc.id not in existing_bm25_doc_ids:
                            # BM25 corpus is missing this document, add chunks to new_bm25_docs
                            # so they get saved and indexed at the end of this function
                            for i, chunk_text in enumerate(temp_chunks):
                                new_bm25_docs.append({
                                    "text": chunk_text,
                                    "metadata": {
                                        "doc_id": doc.id,
                                        "doc_name": doc.filename,
                                        "chunk_idx": i,
                                        "kb_id": kb_id,
                                    }
                                })
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

                    # Add new chunks to BM25 corpus with full metadata
                    for chunk_doc in new_chunk_docs:
                        new_bm25_docs.append({
                            "text": chunk_doc.page_content,
                            "metadata": {
                                "doc_id": chunk_doc.metadata.get("doc_id", ""),
                                "doc_name": chunk_doc.metadata.get("doc_name", ""),
                                "chunk_idx": chunk_doc.metadata.get("chunk_idx", 0),
                                "kb_id": chunk_doc.metadata.get("kb_id", ""),
                            }
                        })

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

        # Build and save BM25 index if there are new chunks
        if new_bm25_docs:
            # Add new chunks to corpus
            bm25_corpus.extend(new_bm25_docs)
            self._save_bm25_corpus(kb_id, bm25_corpus)

            # Rebuild BM25 index
            try:
                from rank_bm25 import BM25Okapi
                # Tokenize texts from corpus with metadata
                tokenized_corpus = [self._tokenize(doc["text"]) for doc in bm25_corpus]
                bm25_index = BM25Okapi(tokenized_corpus)
                self._save_bm25_index(kb_id, bm25_index)
            except ImportError:
                print("Warning: rank-bm25 not installed, BM25 retrieval disabled")

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
        use_hybrid: bool = True,
    ) -> List[Document]:
        """
        Search for relevant chunks in a knowledge base.

        Args:
            kb_id: Knowledge base identifier
            query: Search query
            k: Number of results to retrieve initially
            use_rerank: Whether to use reranker for better results
            top_n: Number of final results after reranking
            use_hybrid: Whether to use hybrid retrieval (vector + BM25)

        Returns:
            List of relevant Document objects
        """
        vectorstore = self._get_vectorstore(kb_id)

        # Get more results for better fusion
        retrieve_k = k * 3 if use_rerank else k

        # Vector search
        vector_results = vectorstore.similarity_search(query, k=retrieve_k)

        # BM25 search (hybrid retrieval)
        bm25_results = []
        if use_hybrid:
            bm25_results = self._bm25_search(kb_id, query, k=retrieve_k)

        # Combine results using RRF (Reciprocal Rank Fusion)
        # Note: Do NOT pass retrieve_k as rrf_k - use default rrf_k=60
        if use_hybrid and bm25_results:
            results = self._rrf_fusion(vector_results, bm25_results)
        else:
            results = vector_results

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

    def _bm25_search(self, kb_id: str, query: str, k: int = 10) -> List[Document]:
        """
        Search using BM25.

        Args:
            kb_id: Knowledge base identifier
            query: Search query
            k: Number of results to retrieve

        Returns:
            List of Document objects ranked by BM25 score with full metadata
        """
        bm25_index = self._load_bm25_index(kb_id)
        if bm25_index is None:
            return []

        bm25_corpus = self._load_bm25_corpus(kb_id)
        if not bm25_corpus:
            return []

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get BM25 scores
        scores = bm25_index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        # Build Document objects with full metadata
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with positive scores
                doc_data = bm25_corpus[idx]
                results.append(
                    Document(
                        page_content=doc_data["text"],
                        metadata={
                            **doc_data.get("metadata", {}),
                            "bm25_score": scores[idx],
                        },
                    )
                )

        return results

    def _rrf_fusion(
        self,
        vector_results: List[Document],
        bm25_results: List[Document],
        rrf_k: int = 60,
    ) -> List[Document]:
        """
        Combine search results using Reciprocal Rank Fusion (RRF).

        RRF formula: score = 1 / (k + rank)
        Default k=60 works well in practice.

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            rrf_k: RRF smoothing constant (default 60, do not confuse with retrieve_k)

        Returns:
            Combined and ranked Document objects
        """
        from collections import defaultdict

        scores: defaultdict = defaultdict(float)
        doc_map: dict = {}

        # Score from vector results
        for rank, doc in enumerate(vector_results):
            # Use doc_id + chunk_idx as unique key (safe identifier)
            doc_id = doc.metadata.get("doc_id", "")
            chunk_idx = doc.metadata.get("chunk_idx", "")
            doc_key = f"{doc_id}_{chunk_idx}"

            # Fallback to content hash if metadata is missing
            if not doc_id:
                import hashlib
                doc_key = hashlib.md5(doc.page_content.encode()).hexdigest()

            scores[doc_key] += 1.0 / (rrf_k + rank)
            doc_map[doc_key] = doc

        # Score from BM25 results
        for rank, doc in enumerate(bm25_results):
            doc_id = doc.metadata.get("doc_id", "")
            chunk_idx = doc.metadata.get("chunk_idx", "")
            doc_key = f"{doc_id}_{chunk_idx}"

            # Fallback to content hash if metadata is missing
            if not doc_id:
                import hashlib
                doc_key = hashlib.md5(doc.page_content.encode()).hexdigest()

            scores[doc_key] += 1.0 / (rrf_k + rank)
            if doc_key not in doc_map:
                doc_map[doc_key] = doc

        # Sort by combined score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Return documents with RRF score
        results = []
        for doc_key, score in sorted_docs:
            doc = doc_map[doc_key]
            doc.metadata["rrf_score"] = score
            results.append(doc)

        return results

    def search_multi(
        self,
        kb_ids: List[str],
        query: str,
        k_per_kb: int = 2,
        use_rerank: bool = False,
        top_n: int = 5,
        use_hybrid: bool = True,
    ) -> List[Document]:
        """
        Search across multiple knowledge bases.

        Args:
            kb_ids: List of knowledge base IDs to search
            query: Search query
            k_per_kb: Number of results per knowledge base
            use_rerank: Whether to use reranker
            top_n: Final number of results after reranking
            use_hybrid: Whether to use hybrid retrieval (vector + BM25)

        Returns:
            List of relevant Document objects
        """
        if use_rerank and settings.RERANKER_MODEL:
            # Collect all texts from all KBs for reranking
            all_texts = []
            all_docs = []

            for kb_id in kb_ids:
                try:
                    results = self.search(kb_id, query, k=k_per_kb * 2, use_hybrid=use_hybrid)
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
                results = self.search(kb_id, query, k=k_per_kb, use_hybrid=use_hybrid)
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
        use_hybrid: bool = True,
    ) -> str:
        """
        Search and return context as a formatted string.

        Args:
            kb_ids: List of knowledge base IDs to search
            query: Search query
            k_per_kb: Number of results per knowledge base
            use_rerank: Whether to use reranker
            top_n: Max number of results
            use_hybrid: Whether to use hybrid retrieval (vector + BM25)

        Returns:
            Formatted context string
        """
        results = self.search_multi(kb_ids, query, k_per_kb, use_rerank, top_n, use_hybrid)

        if not results:
            return ""

        context_parts = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("doc_name", "Unknown")
            # Show RRF score for hybrid, rerank score for reranking
            score = doc.metadata.get("rerank_score") or doc.metadata.get("rrf_score") or doc.metadata.get("bm25_score")
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

        Deletes from:
        - ChunkTracker
        - ChromaDB (vector store)
        - BM25 corpus and index (to prevent "ghost data" leaks)

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
        _ = chunk_tracker.remove_doc_chunks(doc_id)

        # Delete from ChromaDB using metadata filter
        vector_deleted = 0
        try:
            vectorstore = self._get_vectorstore(kb_id)
            if vectorstore:
                # Chroma supports delete by where filter
                deleted = vectorstore.delete(where={"doc_id": doc_id})
                # deleted is a dict with 'ids' key containing deleted IDs
                vector_deleted = len(deleted.get("ids", [])) if deleted else 0
        except Exception as e:
            print(f"Warning: Failed to delete vectors from ChromaDB: {e}")

        # Delete from BM25 corpus and rebuild index
        self._delete_from_bm25(kb_id, doc_id)

        return vector_deleted

    def _delete_from_bm25(self, kb_id: str, doc_id: str):
        """
        Delete a document's chunks from BM25 corpus and rebuild the index.

        Args:
            kb_id: Knowledge base identifier
            doc_id: Document identifier to delete
        """
        bm25_corpus = self._load_bm25_corpus(kb_id)
        if not bm25_corpus:
            return

        # Filter out chunks belonging to this document
        original_count = len(bm25_corpus)
        bm25_corpus = [
            item for item in bm25_corpus
            if item.get("metadata", {}).get("doc_id") != doc_id
        ]

        if len(bm25_corpus) < original_count:
            # Check if corpus is empty after deletion (last document removed)
            if len(bm25_corpus) == 0:
                # Delete BM25 files instead of rebuilding with empty corpus
                self._delete_bm25_index(kb_id)
                return

            # Save updated corpus
            self._save_bm25_corpus(kb_id, bm25_corpus)

            # Rebuild BM25 index
            try:
                from rank_bm25 import BM25Okapi
                tokenized_corpus = [self._tokenize(doc["text"]) for doc in bm25_corpus]
                bm25_index = BM25Okapi(tokenized_corpus)
                self._save_bm25_index(kb_id, bm25_index)
            except ImportError:
                print("Warning: rank-bm25 not installed, BM25 retrieval disabled")
            except Exception as e:
                print(f"Warning: Failed to rebuild BM25 index: {e}")

    def delete_knowledge_base_vectors(self, kb_id: str) -> int:
        """
        Delete all vectors belonging to a knowledge base.

        This deletes:
        - Entire ChromaDB collection for the KB
        - BM25 corpus and index files

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

                # Delete BM25 files
                self._delete_bm25_index(kb_id)

                return count
        except Exception as e:
            print(f"Warning: Failed to delete knowledge base vectors: {e}")
        return 0

    def _delete_bm25_index(self, kb_id: str):
        """
        Delete BM25 corpus and index files for a knowledge base.

        Args:
            kb_id: Knowledge base identifier
        """
        corpus_path = self.bm25_dir / f"kb_{kb_id}_corpus.json"
        index_path = self.bm25_dir / f"kb_{kb_id}_bm25.pkl"

        try:
            if corpus_path.exists():
                corpus_path.unlink()
            if index_path.exists():
                index_path.unlink()
        except Exception as e:
            print(f"Warning: Failed to delete BM25 files: {e}")

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
