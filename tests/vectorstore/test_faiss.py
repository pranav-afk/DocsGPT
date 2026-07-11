import io
import os
from unittest.mock import Mock, patch

import pytest


def _configure_settings(mock_settings, embeddings_name="test_model", index_type="flat"):
    """Apply vector-store settings used by FaissStore unit tests.

    Existing tests default to ``flat`` so they exercise the historical
    ``FAISS.from_documents`` path. HNSW-specific tests override ``index_type``.
    """
    mock_settings.EMBEDDINGS_NAME = embeddings_name
    mock_settings.FAISS_INDEX_TYPE = index_type
    mock_settings.FAISS_HNSW_M = 16
    mock_settings.FAISS_HNSW_EF_CONSTRUCTION = 200
    mock_settings.FAISS_HNSW_EF_SEARCH = 64


@pytest.fixture
def mock_embeddings():
    emb = Mock()
    emb.embed_query = Mock(return_value=[0.1, 0.2, 0.3])
    emb.embed_documents = Mock(return_value=[[0.1, 0.2, 0.3]])
    emb.dimension = 3
    return emb


@pytest.fixture
def mock_storage():
    storage = Mock()
    storage.file_exists = Mock(return_value=True)
    storage.get_file = Mock(return_value=io.BytesIO(b"fake data"))
    storage.save_file = Mock()
    return storage


@pytest.fixture
def mock_docsearch():
    ds = Mock()
    ds.similarity_search = Mock(return_value=[])
    ds.add_texts = Mock(return_value=["id1"])
    ds.add_documents = Mock(return_value=["id1"])
    ds.save_local = Mock()
    ds.delete = Mock()
    ds.index = Mock()
    ds.index.d = 3
    ds.docstore = Mock()
    ds.docstore._dict = {
        "doc1": Mock(page_content="text1", metadata={"source": "a"}),
        "doc2": Mock(page_content="text2", metadata={"source": "b"}),
    }
    return ds


@pytest.mark.unit
class TestFaissStoreInit:
    @patch("application.vectorstore.faiss.StorageCreator")
    @patch("application.vectorstore.faiss.FAISS")
    @patch.object(
        __import__("application.vectorstore.base", fromlist=["BaseVectorStore"]).BaseVectorStore,
        "_get_embeddings",
    )
    @patch("application.vectorstore.faiss.settings")
    def test_init_with_docs(self, mock_settings, mock_get_emb, mock_faiss, mock_storage_creator):
        _configure_settings(mock_settings)
        mock_emb = Mock(dimension=3)
        mock_get_emb.return_value = mock_emb
        mock_ds = Mock()
        mock_ds.index = Mock(d=3)
        mock_faiss.from_documents.return_value = mock_ds
        mock_storage_creator.get_storage.return_value = Mock()

        from application.vectorstore.faiss import FaissStore

        store = FaissStore(source_id="test", embeddings_key="key", docs_init=[Mock()])
        mock_faiss.from_documents.assert_called_once()
        assert store.docsearch is mock_ds

    @patch("application.vectorstore.faiss.StorageCreator")
    @patch("application.vectorstore.faiss.FAISS")
    @patch.object(
        __import__("application.vectorstore.base", fromlist=["BaseVectorStore"]).BaseVectorStore,
        "_get_embeddings",
    )
    @patch("application.vectorstore.faiss.settings")
    def test_init_missing_index_files(
        self, mock_settings, mock_get_emb, mock_faiss, mock_storage_creator
    ):
        _configure_settings(mock_settings)
        mock_emb = Mock(dimension=3)
        mock_get_emb.return_value = mock_emb
        mock_storage = Mock()
        mock_storage.file_exists.return_value = False
        mock_storage_creator.get_storage.return_value = mock_storage

        from application.vectorstore.faiss import FaissStore

        with pytest.raises(Exception, match="Error loading FAISS index"):
            FaissStore(source_id="test", embeddings_key="key")


@pytest.mark.unit
class TestFaissStoreSearch:
    @patch("application.vectorstore.faiss.StorageCreator")
    @patch("application.vectorstore.faiss.FAISS")
    @patch.object(
        __import__("application.vectorstore.base", fromlist=["BaseVectorStore"]).BaseVectorStore,
        "_get_embeddings",
    )
    @patch("application.vectorstore.faiss.settings")
    def test_search_delegates_to_docsearch(
        self, mock_settings, mock_get_emb, mock_faiss, mock_storage_creator
    ):
        _configure_settings(mock_settings)
        mock_emb = Mock(dimension=3)
        mock_get_emb.return_value = mock_emb
        mock_ds = Mock()
        mock_ds.index = Mock(d=3)
        mock_ds.similarity_search.return_value = ["doc1"]
        mock_faiss.from_documents.return_value = mock_ds
        mock_storage_creator.get_storage.return_value = Mock()

        from application.vectorstore.faiss import FaissStore

        store = FaissStore(source_id="t", embeddings_key="k", docs_init=[Mock()])
        result = store.search("query", k=5)
        mock_ds.similarity_search.assert_called_once_with("query", k=5)
        assert result == ["doc1"]

    @patch("application.vectorstore.faiss.StorageCreator")
    @patch("application.vectorstore.faiss.FAISS")
    @patch.object(
        __import__("application.vectorstore.base", fromlist=["BaseVectorStore"]).BaseVectorStore,
        "_get_embeddings",
    )
    @patch("application.vectorstore.faiss.settings")
    def test_search_ignores_score_threshold(
        self, mock_settings, mock_get_emb, mock_faiss, mock_storage_creator
    ):
        # FAISS has no relevance-threshold knob; the per-source score_threshold
        # must be safely dropped, not forwarded (which would crash langchain).
        _configure_settings(mock_settings)
        mock_get_emb.return_value = Mock(dimension=3)
        mock_ds = Mock()
        mock_ds.index = Mock(d=3)
        mock_ds.similarity_search.return_value = ["doc1"]
        mock_faiss.from_documents.return_value = mock_ds
        mock_storage_creator.get_storage.return_value = Mock()

        from application.vectorstore.faiss import FaissStore

        store = FaissStore(source_id="t", embeddings_key="k", docs_init=[Mock()])
        result = store.search("query", k=5, score_threshold=0.9)
        # score_threshold is stripped before the forward.
        mock_ds.similarity_search.assert_called_once_with("query", k=5)
        assert result == ["doc1"]


@pytest.mark.unit
class TestFaissStoreAddTexts:
    @patch("application.vectorstore.faiss.StorageCreator")
    @patch("application.vectorstore.faiss.FAISS")
    @patch.object(
        __import__("application.vectorstore.base", fromlist=["BaseVectorStore"]).BaseVectorStore,
        "_get_embeddings",
    )
    @patch("application.vectorstore.faiss.settings")
    def test_add_texts_delegates(
        self, mock_settings, mock_get_emb, mock_faiss, mock_storage_creator
    ):
        _configure_settings(mock_settings)
        mock_emb = Mock(dimension=3)
        mock_get_emb.return_value = mock_emb
        mock_ds = Mock()
        mock_ds.index = Mock(d=3)
        mock_ds.add_texts.return_value = ["id1", "id2"]
        mock_faiss.from_documents.return_value = mock_ds
        mock_storage_creator.get_storage.return_value = Mock()

        from application.vectorstore.faiss import FaissStore

        store = FaissStore(source_id="t", embeddings_key="k", docs_init=[Mock()])
        result = store.add_texts(["text1", "text2"])
        assert result == ["id1", "id2"]


@pytest.mark.unit
class TestFaissStoreGetChunks:
    @patch("application.vectorstore.faiss.StorageCreator")
    @patch("application.vectorstore.faiss.FAISS")
    @patch.object(
        __import__("application.vectorstore.base", fromlist=["BaseVectorStore"]).BaseVectorStore,
        "_get_embeddings",
    )
    @patch("application.vectorstore.faiss.settings")
    def test_get_chunks(self, mock_settings, mock_get_emb, mock_faiss, mock_storage_creator):
        _configure_settings(mock_settings)
        mock_emb = Mock(dimension=3)
        mock_get_emb.return_value = mock_emb

        doc1 = Mock(page_content="text1", metadata={"source": "a"})
        doc2 = Mock(page_content="text2", metadata={"source": "b"})

        mock_ds = Mock()
        mock_ds.index = Mock(d=3)
        mock_ds.docstore._dict = {"id1": doc1, "id2": doc2}
        mock_faiss.from_documents.return_value = mock_ds
        mock_storage_creator.get_storage.return_value = Mock()

        from application.vectorstore.faiss import FaissStore

        store = FaissStore(source_id="t", embeddings_key="k", docs_init=[Mock()])
        chunks = store.get_chunks()

        assert len(chunks) == 2
        texts = {c["text"] for c in chunks}
        assert texts == {"text1", "text2"}

    @patch("application.vectorstore.faiss.StorageCreator")
    @patch("application.vectorstore.faiss.FAISS")
    @patch.object(
        __import__("application.vectorstore.base", fromlist=["BaseVectorStore"]).BaseVectorStore,
        "_get_embeddings",
    )
    @patch("application.vectorstore.faiss.settings")
    def test_get_chunks_empty(self, mock_settings, mock_get_emb, mock_faiss, mock_storage_creator):
        _configure_settings(mock_settings)
        mock_emb = Mock(dimension=3)
        mock_get_emb.return_value = mock_emb
        mock_ds = Mock()
        mock_ds.index = Mock(d=3)
        mock_ds.docstore._dict = {}
        mock_faiss.from_documents.return_value = mock_ds
        mock_storage_creator.get_storage.return_value = Mock()

        from application.vectorstore.faiss import FaissStore

        store = FaissStore(source_id="t", embeddings_key="k", docs_init=[Mock()])
        assert store.get_chunks() == []


@pytest.mark.unit
class TestFaissStoreSaveLocal:
    @patch("application.vectorstore.faiss.StorageCreator")
    @patch("application.vectorstore.faiss.FAISS")
    @patch.object(
        __import__("application.vectorstore.base", fromlist=["BaseVectorStore"]).BaseVectorStore,
        "_get_embeddings",
    )
    @patch("application.vectorstore.faiss.settings")
    def test_save_local_with_path(
        self, mock_settings, mock_get_emb, mock_faiss, mock_storage_creator
    ):
        _configure_settings(mock_settings)
        mock_emb = Mock(dimension=3)
        mock_get_emb.return_value = mock_emb
        mock_ds = Mock()
        mock_ds.index = Mock(d=3)
        mock_faiss.from_documents.return_value = mock_ds
        mock_storage = Mock()
        mock_storage_creator.get_storage.return_value = mock_storage

        from application.vectorstore.faiss import FaissStore

        store = FaissStore(source_id="t", embeddings_key="k", docs_init=[Mock()])

        # Mock _save_to_storage to avoid file I/O
        store._save_to_storage = Mock(return_value=True)

        with patch("os.makedirs"):
            result = store.save_local(path="/tmp/test_save")

        mock_ds.save_local.assert_called_once_with("/tmp/test_save")
        store._save_to_storage.assert_called_once()
        assert result is True


@pytest.mark.unit
class TestFaissStoreDeleteIndex:
    @patch("application.vectorstore.faiss.StorageCreator")
    @patch("application.vectorstore.faiss.FAISS")
    @patch.object(
        __import__("application.vectorstore.base", fromlist=["BaseVectorStore"]).BaseVectorStore,
        "_get_embeddings",
    )
    @patch("application.vectorstore.faiss.settings")
    def test_delete_index_delegates(
        self, mock_settings, mock_get_emb, mock_faiss, mock_storage_creator
    ):
        _configure_settings(mock_settings)
        mock_emb = Mock(dimension=3)
        mock_get_emb.return_value = mock_emb
        mock_ds = Mock()
        mock_ds.index = Mock(d=3)
        mock_faiss.from_documents.return_value = mock_ds
        mock_storage_creator.get_storage.return_value = Mock()

        from application.vectorstore.faiss import FaissStore

        store = FaissStore(source_id="t", embeddings_key="k", docs_init=[Mock()])
        store.delete_index(["id1"])
        mock_ds.delete.assert_called_once_with(["id1"])


@pytest.mark.unit
class TestFaissStoreAssertEmbeddingDimensions:
    @patch("application.vectorstore.faiss.StorageCreator")
    @patch("application.vectorstore.faiss.FAISS")
    @patch.object(
        __import__("application.vectorstore.base", fromlist=["BaseVectorStore"]).BaseVectorStore,
        "_get_embeddings",
    )
    @patch("application.vectorstore.faiss.settings")
    def test_dimension_mismatch_raises(
        self, mock_settings, mock_get_emb, mock_faiss, mock_storage_creator
    ):
        _configure_settings(
            mock_settings,
            embeddings_name="huggingface_sentence-transformers/all-mpnet-base-v2",
        )
        mock_emb = Mock(dimension=768)
        mock_get_emb.return_value = mock_emb
        mock_ds = Mock()
        mock_ds.index = Mock(d=512)  # Mismatched dimension
        mock_faiss.from_documents.return_value = mock_ds
        mock_storage_creator.get_storage.return_value = Mock()

        from application.vectorstore.faiss import FaissStore

        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            FaissStore(source_id="t", embeddings_key="k", docs_init=[Mock()])

    @patch("application.vectorstore.faiss.StorageCreator")
    @patch("application.vectorstore.faiss.FAISS")
    @patch.object(
        __import__("application.vectorstore.base", fromlist=["BaseVectorStore"]).BaseVectorStore,
        "_get_embeddings",
    )
    @patch("application.vectorstore.faiss.settings")
    def test_missing_dimension_attr_raises(
        self, mock_settings, mock_get_emb, mock_faiss, mock_storage_creator
    ):
        _configure_settings(
            mock_settings,
            embeddings_name="huggingface_sentence-transformers/all-mpnet-base-v2",
        )
        mock_emb = Mock(spec=[])  # No dimension attribute
        mock_get_emb.return_value = mock_emb
        mock_ds = Mock()
        mock_ds.index = Mock(d=768)
        mock_faiss.from_documents.return_value = mock_ds
        mock_storage_creator.get_storage.return_value = Mock()

        from application.vectorstore.faiss import FaissStore

        with pytest.raises(AttributeError, match="dimension"):
            FaissStore(source_id="t", embeddings_key="k", docs_init=[Mock()])


@pytest.mark.unit
class TestFaissStoreDeleteChunk:
    @patch("application.vectorstore.faiss.StorageCreator")
    @patch("application.vectorstore.faiss.FAISS")
    @patch.object(
        __import__("application.vectorstore.base", fromlist=["BaseVectorStore"]).BaseVectorStore,
        "_get_embeddings",
    )
    @patch("application.vectorstore.faiss.settings")
    def test_delete_chunk(self, mock_settings, mock_get_emb, mock_faiss, mock_storage_creator):
        _configure_settings(mock_settings)
        mock_emb = Mock(dimension=3)
        mock_get_emb.return_value = mock_emb
        mock_ds = Mock()
        mock_ds.index = Mock(d=3)
        mock_faiss.from_documents.return_value = mock_ds
        mock_storage = Mock()
        mock_storage_creator.get_storage.return_value = mock_storage

        from application.vectorstore.faiss import FaissStore

        store = FaissStore(source_id="t", embeddings_key="k", docs_init=[Mock()])
        store._save_to_storage = Mock(return_value=True)

        result = store.delete_chunk("chunk_id")
        mock_ds.delete.assert_called_once_with(["chunk_id"])
        store._save_to_storage.assert_called_once()
        assert result is True


@pytest.mark.unit
class TestGetVectorstore:
    def test_with_path(self):
        from application.vectorstore.faiss import get_vectorstore

        assert get_vectorstore("abc123") == os.path.join("indexes", "abc123")

    def test_without_path(self):
        from application.vectorstore.faiss import get_vectorstore

        assert get_vectorstore("") == "indexes"
        assert get_vectorstore(None) == "indexes"

    def test_with_nested_path(self):
        from application.vectorstore.faiss import get_vectorstore

        assert get_vectorstore("user/source123") == os.path.join(
            "indexes", "user", "source123"
        )

    @pytest.mark.parametrize(
        "malicious_path",
        [
            "../outside",
            "../../etc/passwd",
            "nested/../../../outside",
            "/tmp/evil",
            "..\\outside",
            "valid/../../escape",
        ],
    )
    def test_rejects_path_traversal(self, malicious_path):
        from application.vectorstore.faiss import get_vectorstore

        with pytest.raises(ValueError, match="Invalid source_id path"):
            get_vectorstore(malicious_path)

    def test_allows_mongodb_style_ids(self):
        from application.vectorstore.faiss import get_vectorstore

        assert get_vectorstore("65e8f6a8a7a96b1bdad4154f") == os.path.join(
            "indexes", "65e8f6a8a7a96b1bdad4154f"
        )


@pytest.mark.unit
class TestFaissStoreAddChunk:
    @patch("application.vectorstore.faiss.StorageCreator")
    @patch("application.vectorstore.faiss.FAISS")
    @patch.object(
        __import__("application.vectorstore.base", fromlist=["BaseVectorStore"]).BaseVectorStore,
        "_get_embeddings",
    )
    @patch("application.vectorstore.faiss.settings")
    def test_add_chunk_with_metadata(
        self, mock_settings, mock_get_emb, mock_faiss, mock_storage_creator
    ):
        _configure_settings(mock_settings)
        mock_emb = Mock(dimension=3)
        mock_get_emb.return_value = mock_emb
        mock_ds = Mock()
        mock_ds.index = Mock(d=3)
        mock_ds.add_documents.return_value = ["new_id"]
        mock_faiss.from_documents.return_value = mock_ds
        mock_storage = Mock()
        mock_storage_creator.get_storage.return_value = mock_storage

        from application.vectorstore.faiss import FaissStore

        store = FaissStore(source_id="t", embeddings_key="k", docs_init=[Mock()])
        store._save_to_storage = Mock(return_value=True)

        doc_id = store.add_chunk("new text", metadata={"source": "test"})

        assert doc_id == ["new_id"]
        mock_ds.add_documents.assert_called_once()
        store._save_to_storage.assert_called_once()

    @patch("application.vectorstore.faiss.StorageCreator")
    @patch("application.vectorstore.faiss.FAISS")
    @patch.object(
        __import__("application.vectorstore.base", fromlist=["BaseVectorStore"]).BaseVectorStore,
        "_get_embeddings",
    )
    @patch("application.vectorstore.faiss.settings")
    def test_add_chunk_default_metadata(
        self, mock_settings, mock_get_emb, mock_faiss, mock_storage_creator
    ):
        _configure_settings(mock_settings)
        mock_emb = Mock(dimension=3)
        mock_get_emb.return_value = mock_emb
        mock_ds = Mock()
        mock_ds.index = Mock(d=3)
        mock_ds.add_documents.return_value = ["new_id"]
        mock_faiss.from_documents.return_value = mock_ds
        mock_storage = Mock()
        mock_storage_creator.get_storage.return_value = mock_storage

        from application.vectorstore.faiss import FaissStore

        store = FaissStore(source_id="t", embeddings_key="k", docs_init=[Mock()])
        store._save_to_storage = Mock(return_value=True)

        doc_id = store.add_chunk("new text")

        assert doc_id == ["new_id"]


@pytest.mark.unit
class TestFaissStoreSaveLocalNoPath:
    @patch("application.vectorstore.faiss.StorageCreator")
    @patch("application.vectorstore.faiss.FAISS")
    @patch.object(
        __import__("application.vectorstore.base", fromlist=["BaseVectorStore"]).BaseVectorStore,
        "_get_embeddings",
    )
    @patch("application.vectorstore.faiss.settings")
    def test_save_local_without_path(
        self, mock_settings, mock_get_emb, mock_faiss, mock_storage_creator
    ):
        _configure_settings(mock_settings)
        mock_emb = Mock(dimension=3)
        mock_get_emb.return_value = mock_emb
        mock_ds = Mock()
        mock_ds.index = Mock(d=3)
        mock_faiss.from_documents.return_value = mock_ds
        mock_storage = Mock()
        mock_storage_creator.get_storage.return_value = mock_storage

        from application.vectorstore.faiss import FaissStore

        store = FaissStore(source_id="t", embeddings_key="k", docs_init=[Mock()])
        store._save_to_storage = Mock(return_value=True)

        result = store.save_local()

        # Should NOT call docsearch.save_local with a path
        mock_ds.save_local.assert_not_called()
        store._save_to_storage.assert_called_once()
        assert result is True


@pytest.mark.unit
class TestFaissStoreAssertEmbeddingDimensionsMatch:
    @patch("application.vectorstore.faiss.StorageCreator")
    @patch("application.vectorstore.faiss.FAISS")
    @patch.object(
        __import__("application.vectorstore.base", fromlist=["BaseVectorStore"]).BaseVectorStore,
        "_get_embeddings",
    )
    @patch("application.vectorstore.faiss.settings")
    def test_dimension_match_passes(
        self, mock_settings, mock_get_emb, mock_faiss, mock_storage_creator
    ):
        _configure_settings(
            mock_settings,
            embeddings_name="huggingface_sentence-transformers/all-mpnet-base-v2",
        )
        mock_emb = Mock(dimension=768)
        mock_get_emb.return_value = mock_emb
        mock_ds = Mock()
        mock_ds.index = Mock(d=768)  # Matching dimension
        mock_faiss.from_documents.return_value = mock_ds
        mock_storage_creator.get_storage.return_value = Mock()

        from application.vectorstore.faiss import FaissStore

        # Should not raise
        store = FaissStore(source_id="t", embeddings_key="k", docs_init=[Mock()])
        assert store is not None

    @patch("application.vectorstore.faiss.StorageCreator")
    @patch("application.vectorstore.faiss.FAISS")
    @patch.object(
        __import__("application.vectorstore.base", fromlist=["BaseVectorStore"]).BaseVectorStore,
        "_get_embeddings",
    )
    @patch("application.vectorstore.faiss.settings")
    def test_non_huggingface_skips_dimension_check(
        self, mock_settings, mock_get_emb, mock_faiss, mock_storage_creator
    ):
        _configure_settings(
            mock_settings,
            embeddings_name="openai_text-embedding-ada-002",
        )
        mock_emb = Mock(dimension=1536)
        mock_get_emb.return_value = mock_emb
        mock_ds = Mock()
        mock_ds.index = Mock(d=999)  # Mismatched but doesn't matter
        mock_faiss.from_documents.return_value = mock_ds
        mock_storage_creator.get_storage.return_value = Mock()

        from application.vectorstore.faiss import FaissStore

        # Should not raise since embedding name is not the huggingface one
        store = FaissStore(source_id="t", embeddings_key="k", docs_init=[Mock()])
        assert store is not None


@pytest.mark.unit
class TestFaissHnswIndex:
    """HNSW-specific construction, search params, and delete-via-rebuild."""

    @patch("application.vectorstore.faiss.StorageCreator")
    @patch("application.vectorstore.faiss.FAISS")
    @patch("application.vectorstore.faiss.build_faiss_index")
    @patch("application.vectorstore.faiss.InMemoryDocstore")
    @patch.object(
        __import__("application.vectorstore.base", fromlist=["BaseVectorStore"]).BaseVectorStore,
        "_get_embeddings",
    )
    @patch("application.vectorstore.faiss.settings")
    def test_init_with_docs_uses_hnsw_index(
        self,
        mock_settings,
        mock_get_emb,
        mock_docstore,
        mock_build_index,
        mock_faiss,
        mock_storage_creator,
    ):
        _configure_settings(mock_settings, index_type="hnsw")
        mock_emb = Mock(dimension=3)
        mock_get_emb.return_value = mock_emb
        mock_index = Mock()
        mock_index.d = 3
        mock_build_index.return_value = mock_index
        mock_store = Mock()
        mock_store.index = mock_index
        mock_faiss.return_value = mock_store
        mock_storage_creator.get_storage.return_value = Mock()

        from application.vectorstore.faiss import FaissStore

        store = FaissStore(source_id="t", embeddings_key="k", docs_init=[Mock()])

        mock_build_index.assert_called_once_with(3)
        mock_faiss.assert_called_once()
        mock_store.add_documents.assert_called_once()
        mock_faiss.from_documents.assert_not_called()
        assert store.docsearch is mock_store

    @patch("application.vectorstore.faiss.StorageCreator")
    @patch("application.vectorstore.faiss.FAISS")
    @patch("application.vectorstore.faiss.apply_hnsw_search_params")
    @patch.object(
        __import__("application.vectorstore.base", fromlist=["BaseVectorStore"]).BaseVectorStore,
        "_get_embeddings",
    )
    @patch("application.vectorstore.faiss.settings")
    def test_load_local_applies_ef_search(
        self,
        mock_settings,
        mock_get_emb,
        mock_apply_ef,
        mock_faiss,
        mock_storage_creator,
    ):
        _configure_settings(mock_settings, index_type="hnsw")
        mock_get_emb.return_value = Mock(dimension=3)
        mock_storage = Mock()
        mock_storage.file_exists.return_value = True
        mock_storage.get_file.return_value = io.BytesIO(b"fake")
        mock_storage_creator.get_storage.return_value = mock_storage
        mock_ds = Mock()
        mock_ds.index = Mock(d=3)
        mock_faiss.load_local.return_value = mock_ds

        from application.vectorstore.faiss import FaissStore

        FaissStore(source_id="t", embeddings_key="k")
        mock_apply_ef.assert_called_once_with(mock_ds.index)

    @patch("application.vectorstore.faiss.StorageCreator")
    @patch("application.vectorstore.faiss.FAISS")
    @patch("application.vectorstore.faiss.build_faiss_index")
    @patch("application.vectorstore.faiss.InMemoryDocstore")
    @patch.object(
        __import__("application.vectorstore.base", fromlist=["BaseVectorStore"]).BaseVectorStore,
        "_get_embeddings",
    )
    @patch("application.vectorstore.faiss.settings")
    def test_hnsw_delete_rebuilds_without_id(
        self,
        mock_settings,
        mock_get_emb,
        mock_docstore,
        mock_build_index,
        mock_faiss,
        mock_storage_creator,
    ):
        _configure_settings(mock_settings, index_type="hnsw")
        mock_get_emb.return_value = Mock(dimension=3)
        mock_storage_creator.get_storage.return_value = Mock()

        keep_doc = Mock(page_content="keep", metadata={})
        drop_doc = Mock(page_content="drop", metadata={})

        faiss = pytest.importorskip("faiss")
        hnsw_index = faiss.IndexHNSWFlat(3, 16)
        hnsw_index.hnsw.efConstruction = 200
        hnsw_index.hnsw.efSearch = 64

        original = Mock()
        original.index = hnsw_index
        original.docstore._dict = {"keep": keep_doc, "drop": drop_doc}
        original.delete = Mock(side_effect=AssertionError("native delete must not run"))

        rebuilt = Mock()
        rebuilt.index = Mock(d=3)
        mock_build_index.return_value = Mock(d=3)
        mock_faiss.side_effect = [original, rebuilt]

        from application.vectorstore.faiss import FaissStore

        store = FaissStore(source_id="t", embeddings_key="k", docs_init=[Mock()])
        store.docsearch = original

        store.delete_index(["drop"])

        mock_build_index.assert_called()
        rebuilt.add_documents.assert_called_once()
        args, kwargs = rebuilt.add_documents.call_args
        assert args[0] == [keep_doc]
        assert kwargs.get("ids") == ["keep"]
        assert store.docsearch is rebuilt

    def test_build_faiss_index_hnsw_sets_params(self):
        faiss = pytest.importorskip("faiss")
        with patch("application.vectorstore.faiss.settings") as mock_settings:
            _configure_settings(mock_settings, index_type="hnsw")
            mock_settings.FAISS_HNSW_M = 16
            mock_settings.FAISS_HNSW_EF_CONSTRUCTION = 200
            mock_settings.FAISS_HNSW_EF_SEARCH = 64

            from application.vectorstore.faiss import build_faiss_index

            index = build_faiss_index(8)
            assert isinstance(index, faiss.IndexHNSWFlat)
            assert index.d == 8
            assert index.hnsw.efConstruction == 200
            assert index.hnsw.efSearch == 64

    def test_build_faiss_index_flat(self):
        faiss = pytest.importorskip("faiss")
        with patch("application.vectorstore.faiss.settings") as mock_settings:
            _configure_settings(mock_settings, index_type="flat")

            from application.vectorstore.faiss import build_faiss_index

            index = build_faiss_index(4)
            assert isinstance(index, faiss.IndexFlatL2)
            assert index.d == 4

    def test_build_faiss_index_rejects_unknown_type(self):
        with patch("application.vectorstore.faiss.settings") as mock_settings:
            mock_settings.FAISS_INDEX_TYPE = "ivf"
            from application.vectorstore.faiss import build_faiss_index

            with pytest.raises(ValueError, match="Unsupported FAISS_INDEX_TYPE"):
                build_faiss_index(4)
