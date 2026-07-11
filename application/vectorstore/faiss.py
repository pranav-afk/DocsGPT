import logging
import os
import tempfile
import io
from typing import Any, List, Optional, Sequence

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from application.core.settings import settings
from application.parser.schema.base import Document
from application.vectorstore.base import BaseVectorStore
from application.storage.storage_creator import StorageCreator

logger = logging.getLogger(__name__)


def get_vectorstore(path: str) -> str:
    """Build a safe local path for a FAISS index.

    Args:
        path: Source identifier provided by the caller.

    Returns:
        The validated vectorstore path rooted under ``indexes``.

    Raises:
        ValueError: If ``path`` escapes the ``indexes`` directory.
    """
    base_dir = "indexes"
    if not path:
        return base_dir

    normalized = str(path).strip()
    if "\\" in normalized:
        raise ValueError("Invalid source_id path")

    candidate = os.path.normpath(os.path.join(base_dir, normalized))
    base_abs = os.path.abspath(base_dir)
    candidate_abs = os.path.abspath(candidate)

    if not candidate_abs.startswith(base_abs + os.sep) and candidate_abs != base_abs:
        raise ValueError("Invalid source_id path")

    return candidate


def _resolve_index_type() -> str:
    """Return the configured FAISS index type (``hnsw`` or ``flat``)."""
    index_type = (settings.FAISS_INDEX_TYPE or "hnsw").strip().lower()
    if index_type not in {"hnsw", "flat"}:
        raise ValueError(
            f"Unsupported FAISS_INDEX_TYPE={index_type!r}; expected 'hnsw' or 'flat'"
        )
    return index_type


def build_faiss_index(dimension: int) -> Any:
    """Create a FAISS index for the configured algorithm.

    Args:
        dimension: Embedding dimensionality of vectors that will be stored.

    Returns:
        A FAISS index instance (``IndexHNSWFlat`` or ``IndexFlatL2``).
    """
    if dimension <= 0:
        raise ValueError(f"Embedding dimension must be positive, got {dimension}")

    index_type = _resolve_index_type()
    if index_type == "flat":
        return faiss.IndexFlatL2(dimension)

    m = max(1, int(settings.FAISS_HNSW_M))
    ef_construction = max(1, int(settings.FAISS_HNSW_EF_CONSTRUCTION))
    ef_search = max(1, int(settings.FAISS_HNSW_EF_SEARCH))

    # IndexHNSWFlat stores full vectors and builds an HNSW graph for ANN search.
    index = faiss.IndexHNSWFlat(dimension, m)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    logger.info(
        "Built FAISS HNSW index dim=%s M=%s efConstruction=%s efSearch=%s",
        dimension,
        m,
        ef_construction,
        ef_search,
    )
    return index


def _is_hnsw_index(index: Any) -> bool:
    """Return True when ``index`` is a FAISS HNSW index.

    Uses ``isinstance`` / type name rather than ``hasattr(..., "hnsw")`` so
    unit-test ``Mock`` objects are not treated as HNSW (Mocks auto-create
    attributes on access).
    """
    if index is None:
        return False
    if isinstance(index, faiss.IndexHNSWFlat):
        return True
    # Cover SWIG proxy type-name variants without treating Mock as HNSW.
    name = getattr(type(index), "__name__", "")
    return name == "IndexHNSWFlat"


def apply_hnsw_search_params(index: Any) -> None:
    """Apply runtime ``efSearch`` to a loaded HNSW index when present.

    Args:
        index: FAISS index loaded from storage or newly created.
    """
    if not _is_hnsw_index(index):
        return
    ef_search = max(1, int(settings.FAISS_HNSW_EF_SEARCH))
    index.hnsw.efSearch = ef_search


class FaissStore(BaseVectorStore):
    def __init__(self, source_id: str, embeddings_key: str, docs_init=None):
        super().__init__()
        self.source_id = source_id
        self.path = get_vectorstore(source_id)
        self.embeddings = self._get_embeddings(settings.EMBEDDINGS_NAME, embeddings_key)
        self.storage = StorageCreator.get_storage()

        try:
            if docs_init:
                self.docsearch = self._create_from_documents(docs_init)
            else:
                with tempfile.TemporaryDirectory() as temp_dir:
                    faiss_path = f"{self.path}/index.faiss"
                    pkl_path = f"{self.path}/index.pkl"

                    if not self.storage.file_exists(
                        faiss_path
                    ) or not self.storage.file_exists(pkl_path):
                        raise FileNotFoundError(
                            f"Index files not found in storage at {self.path}"
                        )

                    faiss_file = self.storage.get_file(faiss_path)
                    pkl_file = self.storage.get_file(pkl_path)

                    local_faiss_path = os.path.join(temp_dir, "index.faiss")
                    local_pkl_path = os.path.join(temp_dir, "index.pkl")

                    with open(local_faiss_path, "wb") as f:
                        f.write(faiss_file.read())

                    with open(local_pkl_path, "wb") as f:
                        f.write(pkl_file.read())

                    self.docsearch = FAISS.load_local(
                        temp_dir, self.embeddings, allow_dangerous_deserialization=True
                    )
                    apply_hnsw_search_params(self.docsearch.index)
        except Exception as e:
            raise Exception(f"Error loading FAISS index: {str(e)}")

        self.assert_embedding_dimensions(self.embeddings)

    def _embedding_dimension(self, docs: Optional[Sequence] = None) -> int:
        """Resolve embedding dimension from the model or a sample document."""
        dimension = getattr(self.embeddings, "dimension", None)
        if dimension:
            return int(dimension)
        if docs:
            sample = docs[0]
            text = getattr(sample, "page_content", None)
            if text is None and hasattr(sample, "text"):
                text = sample.text
            if text is None:
                text = str(sample)
            return len(self.embeddings.embed_query(text))
        raise ValueError(
            "Cannot determine embedding dimension without embeddings.dimension "
            "or seed documents"
        )

    def _create_from_documents(self, docs: Sequence) -> FAISS:
        """Build a FAISS store from documents using the configured index type.

        Args:
            docs: Seed documents (LangChain ``Document`` instances).

        Returns:
            A populated LangChain ``FAISS`` vector store.
        """
        index_type = _resolve_index_type()
        if index_type == "flat":
            # Preserve historical exact-search path.
            return FAISS.from_documents(docs, self.embeddings)

        dimension = self._embedding_dimension(docs)
        index = build_faiss_index(dimension)
        store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        if docs:
            store.add_documents(list(docs))
        return store

    def _rebuild_without_ids(self, ids_to_delete: Sequence[str]) -> None:
        """Rebuild the FAISS index excluding the given docstore ids.

        HNSW indexes do not support ``remove_ids``; the standard workaround is
        to soft-omit vectors by rebuilding from the remaining docstore entries.
        """
        delete_set = set(ids_to_delete)
        remaining_docs = []
        remaining_ids: List[str] = []
        docstore_dict = getattr(self.docsearch.docstore, "_dict", {}) or {}
        for doc_id, doc in docstore_dict.items():
            if doc_id not in delete_set:
                remaining_docs.append(doc)
                remaining_ids.append(doc_id)

        dimension = self.docsearch.index.d
        index = build_faiss_index(dimension)
        store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        if remaining_docs:
            store.add_documents(remaining_docs, ids=remaining_ids)
        self.docsearch = store

    def search(self, *args, **kwargs):
        # FAISS has no relevance-threshold knobs; drop it so the per-source
        # score_threshold is safely ignored rather than crashing the forward.
        kwargs.pop("score_threshold", None)
        return self.docsearch.similarity_search(*args, **kwargs)

    def add_texts(self, *args, **kwargs):
        return self.docsearch.add_texts(*args, **kwargs)

    def _save_to_storage(self):
        """
        Save the FAISS index to storage using temporary directory pattern.
        Works consistently for both local and S3 storage.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            self.docsearch.save_local(temp_dir)

            faiss_path = os.path.join(temp_dir, "index.faiss")
            pkl_path = os.path.join(temp_dir, "index.pkl")

            with open(faiss_path, "rb") as f_faiss:
                faiss_data = f_faiss.read()

            with open(pkl_path, "rb") as f_pkl:
                pkl_data = f_pkl.read()

            storage_path = get_vectorstore(self.source_id)
            self.storage.save_file(io.BytesIO(faiss_data), f"{storage_path}/index.faiss")
            self.storage.save_file(io.BytesIO(pkl_data), f"{storage_path}/index.pkl")

        return True

    def save_local(self, path=None):
        if path:
            os.makedirs(path, exist_ok=True)
            self.docsearch.save_local(path)

        self._save_to_storage()

        return True

    def delete_index(self, *args, **kwargs):
        """Delete vectors by docstore id.

        HNSW does not support in-place removal, so those indexes are rebuilt
        without the deleted ids. Flat indexes use FAISS native ``remove_ids``.
        """
        ids = args[0] if args else kwargs.get("ids")
        if ids is not None and _is_hnsw_index(self.docsearch.index):
            self._rebuild_without_ids(list(ids))
            return True
        return self.docsearch.delete(*args, **kwargs)

    def assert_embedding_dimensions(self, embeddings):
        """Check that the word embedding dimension of the docsearch index matches the dimension of the word embeddings used."""
        if (
            settings.EMBEDDINGS_NAME
            == "huggingface_sentence-transformers/all-mpnet-base-v2"
        ):
            word_embedding_dimension = getattr(embeddings, "dimension", None)
            if word_embedding_dimension is None:
                raise AttributeError(
                    "'dimension' attribute not found in embeddings instance."
                )

            docsearch_index_dimension = self.docsearch.index.d
            if word_embedding_dimension != docsearch_index_dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: embeddings.dimension ({word_embedding_dimension}) != docsearch index dimension ({docsearch_index_dimension})"
                )

    def get_chunks(self):
        chunks = []
        if self.docsearch:
            for doc_id, doc in self.docsearch.docstore._dict.items():
                chunk_data = {
                    "doc_id": doc_id,
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                }
                chunks.append(chunk_data)
        return chunks

    def add_chunk(self, text, metadata=None):
        """Add a new chunk and save to storage."""
        metadata = metadata or {}
        doc = Document(text=text, extra_info=metadata).to_langchain_format()
        doc_id = self.docsearch.add_documents([doc])
        self._save_to_storage()
        return doc_id

    def delete_chunk(self, chunk_id):
        """Delete a chunk and save to storage."""
        self.delete_index([chunk_id])
        self._save_to_storage()
        return True
