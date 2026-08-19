"""Tests for the NeMo Retriever document_search tool.

These exercise the tool's own logic — gating, path resolution, argument
validation, hit normalization, and the JSON envelope — against a temp
HERMES_HOME with the heavy ``nemo-retriever`` SDK fully stubbed. The single
SDK-touching entry point (:func:`_nemo_retriever_search`) is monkeypatched so
the suite never needs the real package, a GPU, or an NVIDIA API key.
"""

import json

import pytest

from tools import nemo_retriever_tool as nrt


# ---------------------------------------------------------------------------
# check_fn gating
# ---------------------------------------------------------------------------

def test_gate_false_without_credential_or_local(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr(nrt, "_load_config", lambda: {})
    assert nrt.check_document_search_requirements() is False


def test_gate_true_with_api_key(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setattr(nrt, "_load_config", lambda: {})
    assert nrt.check_document_search_requirements() is True


def test_gate_true_with_local_flag(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr(nrt, "_load_config", lambda: {"local": True})
    assert nrt.check_document_search_requirements() is True


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def test_collect_documents_filters_supported(tmp_path):
    pdf = tmp_path / "report.pdf"
    docx = tmp_path / "notes.docx"
    html = tmp_path / "page.html"
    code = tmp_path / "main.py"
    for p in (pdf, docx, html, code):
        p.write_text("x", encoding="utf-8")

    files, skipped = nrt._collect_documents([str(pdf), str(docx), str(html), str(code)])

    assert set(files) == {str(pdf.resolve()), str(docx.resolve()), str(html.resolve())}
    assert any("main.py" in s and "unsupported" in s for s in skipped)


def test_collect_documents_walks_directory_and_skips_dotdirs(tmp_path):
    (tmp_path / "a.pdf").write_text("x", encoding="utf-8")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "b.htm").write_text("x", encoding="utf-8")
    hidden = tmp_path / ".git"
    hidden.mkdir()
    (hidden / "c.pdf").write_text("x", encoding="utf-8")

    files, _ = nrt._collect_documents([str(tmp_path)])

    names = {p.rsplit("/", 1)[-1] for p in files}
    assert names == {"a.pdf", "b.htm"}  # .git/c.pdf excluded


def test_collect_documents_reports_missing(tmp_path):
    files, skipped = nrt._collect_documents([str(tmp_path / "nope.pdf")])
    assert files == []
    assert any("not found" in s for s in skipped)


def test_derive_table_name_stable_and_named(tmp_path):
    files = [str(tmp_path / "z.pdf"), str(tmp_path / "a.pdf")]
    # Order-independent (sorted internally)
    assert nrt._derive_table_name(files, None) == nrt._derive_table_name(list(reversed(files)), None)
    # Explicit name wins and is sanitized
    assert nrt._derive_table_name(files, "My Corpus!") == "docs_My_Corpus_"


# ---------------------------------------------------------------------------
# Argument validation / envelope
# ---------------------------------------------------------------------------

def _enable(monkeypatch, cfg=None):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setattr(nrt, "_load_config", lambda: cfg or {})


def test_empty_query_errors(monkeypatch):
    _enable(monkeypatch)
    out = json.loads(nrt.document_search(query="  ", paths=["x.pdf"]))
    assert "query" in out["error"]


def test_missing_paths_errors(monkeypatch):
    _enable(monkeypatch)
    out = json.loads(nrt.document_search(query="hi", paths=None))
    assert "paths" in out["error"]


def test_not_configured_errors(monkeypatch, tmp_path):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr(nrt, "_load_config", lambda: {})
    pdf = tmp_path / "a.pdf"
    pdf.write_text("x", encoding="utf-8")
    out = json.loads(nrt.document_search(query="hi", paths=[str(pdf)]))
    assert "not configured" in out["error"]
    assert "NVIDIA_API_KEY" in out["error"]


def test_no_supported_files_errors(monkeypatch, tmp_path):
    _enable(monkeypatch)
    code = tmp_path / "main.py"
    code.write_text("x", encoding="utf-8")
    out = json.loads(nrt.document_search(query="hi", paths=[str(code)]))
    assert "No PDF" in out["error"]
    assert out.get("skipped")


def test_happy_path_envelope(monkeypatch, tmp_path):
    _enable(monkeypatch, {"top_k": 5, "hybrid": True})
    pdf = tmp_path / "report.pdf"
    pdf.write_text("x", encoding="utf-8")

    captured = {}

    def _fake_search(query, files, uri, table_name, top_k, reindex, cfg):
        captured.update(
            query=query, files=files, top_k=top_k, reindex=reindex, table_name=table_name
        )
        hits = [
            {"text": "alpha passage", "score": 0.9, "metadata": {"source_metadata": {"source_name": "report.pdf"}, "page": 2}},
            {"text": "beta passage", "_distance": 0.3},
        ]
        return hits, True

    monkeypatch.setattr(nrt, "_nemo_retriever_search", _fake_search)

    out = json.loads(nrt.document_search(query="find alpha", paths=[str(pdf)], top_k=2))

    assert out["success"] is True
    assert out["backend"] == "nemo_retriever"
    assert out["retrieval"] == "dense"
    assert out["reindexed"] is True
    assert out["count"] == 2
    assert out["hits"][0]["text"] == "alpha passage"
    assert out["hits"][0]["source"] == "report.pdf"
    assert out["hits"][0]["page"] == 2
    assert out["hits"][1]["score"] == 0.3  # _distance alias
    assert captured["top_k"] == 2
    assert captured["files"] == [str(pdf.resolve())]


def test_top_k_clamped_and_defaulted(monkeypatch, tmp_path):
    _enable(monkeypatch, {"top_k": 7})
    pdf = tmp_path / "a.pdf"
    pdf.write_text("x", encoding="utf-8")

    seen = {}

    def _fake_search(query, files, uri, table_name, top_k, reindex, cfg):
        seen["top_k"] = top_k
        return [], False

    monkeypatch.setattr(nrt, "_nemo_retriever_search", _fake_search)

    # No top_k → config default (7)
    nrt.document_search(query="q", paths=[str(pdf)])
    assert seen["top_k"] == 7

    # Over the cap → clamped to 50
    nrt.document_search(query="q", paths=[str(pdf)], top_k=999)
    assert seen["top_k"] == 50


def test_dense_retrieval_label_when_hybrid_disabled(monkeypatch, tmp_path):
    _enable(monkeypatch, {"hybrid": False})
    pdf = tmp_path / "a.pdf"
    pdf.write_text("x", encoding="utf-8")
    monkeypatch.setattr(nrt, "_nemo_retriever_search", lambda **k: ([], False))
    out = json.loads(nrt.document_search(query="q", paths=[str(pdf)]))
    assert out["retrieval"] == "dense"


def test_query_hybrid_forced_off_on_pinned_sdk():
    """Pinned nemo-retriever 26.5.0 cannot hybrid-query precomputed vectors."""
    assert nrt._query_hybrid_enabled({"hybrid": True}) is False
    assert nrt._query_hybrid_enabled({"hybrid": False}) is False
    assert nrt._query_hybrid_enabled({}) is False


def test_resolve_embedding_endpoint_local_skips_cloud_default():
    assert nrt._resolve_embedding_endpoint({"local": True}) is None
    assert nrt._resolve_embedding_endpoint({"local": True, "embedding_endpoint": ""}) is None
    assert nrt._resolve_embedding_endpoint({"local": True, "embedding_endpoint": "none"}) is None
    assert nrt._resolve_embedding_endpoint({"local": True, "embedding_endpoint": "hf"}) is None
    assert (
        nrt._resolve_embedding_endpoint(
            {"local": True, "embedding_endpoint": "https://example.com/v1/embeddings"}
        )
        is None
    )


def test_resolve_embedding_endpoint_remote_default_when_not_local():
    assert (
        nrt._resolve_embedding_endpoint({"local": False})
        == nrt.DEFAULT_EMBEDDING_ENDPOINT
    )
    assert (
        nrt._resolve_embedding_endpoint(
            {"local": False, "embedding_endpoint": "https://example.com/v1/embeddings"}
        )
        == "https://example.com/v1/embeddings"
    )


def test_embed_kwargs_local_hf_omits_endpoint():
    kw = nrt._embed_kwargs(
        {
            "local": True,
            "embedding_model": "nvidia/llama-nemotron-embed-1b-v2",
            "local_ingest_embed_backend": "hf",
            "local_hf_device": "cuda:0",
        }
    )
    assert "embedding_endpoint" not in kw
    assert "embed_invoke_url" not in kw
    assert "local_hf_device" not in kw
    assert kw["model_name"] == "nvidia/llama-nemotron-embed-1b-v2"
    assert kw["local_ingest_embed_backend"] == "hf"
    assert kw["runtime"]["device"] == "cuda:0"
    assert nrt._is_local_mode(
        {"local": True, "embedding_model": "nvidia/llama-nemotron-embed-1b-v2"}
    )


def test_embed_kwargs_remote_includes_endpoint():
    kw = nrt._embed_kwargs(
        {
            "local": False,
            "embedding_endpoint": "https://integrate.api.nvidia.com/v1/embeddings",
            "embedding_model": "nvidia/llama-nemotron-embed-1b-v2",
        }
    )
    assert kw["embedding_endpoint"].startswith("https://")
    assert kw["embed_invoke_url"] == kw["embedding_endpoint"]
    assert "local_ingest_embed_backend" not in kw


def test_extract_kwargs_local_disables_remote_page_elements():
    kw = nrt._extract_kwargs(
        {
            "local": True,
            "use_page_elements": True,
            "extract_tables": True,
            "extract_charts": True,
        }
    )
    assert kw["method"] == "pdfium"
    assert kw["use_page_elements"] is False
    assert kw["extract_tables"] is False
    assert kw["extract_charts"] is False


def test_ingest_documents_local_passes_hf_embed_kwargs(monkeypatch, tmp_path):
    calls = {}

    class _FakeIngestor:
        def files(self, files):
            return self

        def extract(self, **kw):
            calls["extract"] = kw
            return self

        def embed(self, **kw):
            calls["embed"] = kw
            return self

        def vdb_upload(self, params=None, **kwargs):
            return self

        def ingest(self):
            calls["ingest"] = True

    import sys
    import types

    fake_nr = types.ModuleType("nemo_retriever")
    fake_nr.create_ingestor = lambda **kw: _FakeIngestor()
    fake_params = types.ModuleType("nemo_retriever.params")

    class _FakeEmbedParams:
        @classmethod
        def model_validate(cls, data):
            calls["validated"] = data
            return data

    fake_params.EmbedParams = _FakeEmbedParams
    monkeypatch.setitem(sys.modules, "nemo_retriever", fake_nr)
    monkeypatch.setitem(sys.modules, "nemo_retriever.params", fake_params)
    monkeypatch.setattr(nrt, "_verify_local_embedder", lambda cfg: None)

    pdf = tmp_path / "a.pdf"
    pdf.write_text("x", encoding="utf-8")
    nrt._ingest_documents(
        [str(pdf)],
        tmp_path / "idx",
        "docs_test",
        {"local": True, "embedding_endpoint": "", "local_ingest_embed_backend": "hf"},
    )

    assert calls.get("ingest") is True
    embed = calls.get("embed") or {}
    validated = calls.get("validated") or {}
    assert "embed_invoke_url" not in validated
    assert "embedding_endpoint" not in validated
    assert validated.get("local_ingest_embed_backend") == "hf"
    assert validated.get("runtime", {}).get("device") == "cuda:0"
    assert embed.get("params") is validated
    extract = calls.get("extract") or {}
    assert extract.get("use_page_elements") is False


def _install_fake_embedder_module(monkeypatch, factory):
    import sys
    import types

    fake_nr = types.ModuleType("nemo_retriever")
    fake_nr.__path__ = []  # mark as package so `nemo_retriever.model` resolves
    fake_model = types.ModuleType("nemo_retriever.model")
    fake_model.create_local_embedder = factory
    monkeypatch.setitem(sys.modules, "nemo_retriever", fake_nr)
    monkeypatch.setitem(sys.modules, "nemo_retriever.model", fake_model)


def test_require_local_cuda_rejects_unavailable_gpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="No remote fallback was attempted"):
        nrt._require_local_cuda({"local": True, "local_hf_device": "cuda:0"})


def test_require_local_cuda_accepts_configured_gpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert (
        nrt._require_local_cuda({"local": True, "local_hf_device": "cuda:0"})
        == "cuda:0"
    )


def test_require_local_cuda_rejects_cpu_device():
    with pytest.raises(RuntimeError, match="strict local GPU mode"):
        nrt._require_local_cuda({"local": True, "local_hf_device": "cpu"})


def test_verify_local_embedder_raises_actionable_error(monkeypatch):
    """A broken local embedder must fail up front, not silently embed nothing."""

    def _boom(*a, **kw):
        raise ModuleNotFoundError("No module named 'transformers'")

    _install_fake_embedder_module(monkeypatch, _boom)
    monkeypatch.setattr(nrt, "_require_local_cuda", lambda cfg: "cuda:0")

    with pytest.raises(RuntimeError) as excinfo:
        nrt._verify_local_embedder(
            {"local": True, "embedding_endpoint": "", "local_ingest_embed_backend": "hf"}
        )

    msg = str(excinfo.value)
    assert "transformers" in msg
    assert "embedding_endpoint" in msg


def test_verify_local_embedder_accepts_vectors(monkeypatch):
    class _Embedder:
        def embed(self, texts, batch_size=1):
            return [[0.1, 0.2, 0.3] for _ in texts]

    _install_fake_embedder_module(monkeypatch, lambda *a, **kw: _Embedder())
    monkeypatch.setattr(nrt, "_require_local_cuda", lambda cfg: "cuda:0")

    nrt._verify_local_embedder(
        {"local": True, "embedding_endpoint": "", "local_ingest_embed_backend": "hf"}
    )


def test_query_index_falls_back_to_dense_on_hybrid_not_implemented(monkeypatch, tmp_path):
    """If hybrid somehow gets enabled and SDK raises, retry dense."""
    import sys
    import types

    calls = {"n": 0}

    class _FakeRetriever:
        def __init__(self, **kwargs):
            self.vdb_kwargs = kwargs.get("vdb_kwargs") or {}

        def query(self, query):
            calls["n"] += 1
            if self.vdb_kwargs.get("hybrid"):
                raise NotImplementedError(
                    "LanceDB hybrid retrieval with precomputed vectors is not implemented yet."
                )
            return [{"text": "hit", "score": 1.0}]

    monkeypatch.setattr(nrt, "_query_hybrid_enabled", lambda cfg: True)
    if "nemo_retriever" not in sys.modules:
        monkeypatch.setitem(sys.modules, "nemo_retriever", types.ModuleType("nemo_retriever"))
    mod = types.ModuleType("nemo_retriever.retriever")
    mod.Retriever = _FakeRetriever
    monkeypatch.setitem(sys.modules, "nemo_retriever.retriever", mod)

    hits = nrt._query_index("coal", tmp_path, "docs_t", 3, {"hybrid": True})
    assert hits == [{"text": "hit", "score": 1.0}]
    assert calls["n"] == 2  # hybrid attempt + dense retry


def test_sdk_runtime_error_becomes_tool_error(monkeypatch, tmp_path):
    _enable(monkeypatch)
    pdf = tmp_path / "a.pdf"
    pdf.write_text("x", encoding="utf-8")

    def _boom(query, files, uri, table_name, top_k, reindex, cfg):
        raise RuntimeError("SDK not installed; requires Python 3.12")

    monkeypatch.setattr(nrt, "_nemo_retriever_search", _boom)
    out = json.loads(nrt.document_search(query="q", paths=[str(pdf)]))
    assert "Python 3.12" in out["error"]


def test_max_chars_truncation(monkeypatch, tmp_path):
    _enable(monkeypatch, {"max_chars": 10})
    pdf = tmp_path / "a.pdf"
    pdf.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        nrt, "_nemo_retriever_search",
        lambda **k: ([{"text": "x" * 100}], False),
    )
    out = json.loads(nrt.document_search(query="q", paths=[str(pdf)]))
    assert out["hits"][0]["text"].endswith("...[truncated]")
    assert len(out["hits"][0]["text"]) < 100


# ---------------------------------------------------------------------------
# Hit normalization
# ---------------------------------------------------------------------------

def test_normalize_hit_non_dict():
    h = nrt._normalize_hit("plain string", max_chars=100)
    assert h == {"text": "plain string", "score": None, "source": None, "page": None}


def test_normalize_hit_content_alias():
    h = nrt._normalize_hit({"content": "body", "_relevance_score": 1.5}, max_chars=100)
    assert h["text"] == "body"
    assert h["score"] == 1.5


# ---------------------------------------------------------------------------
# Ingest API contract (nemo-retriever GraphIngestor)
# ---------------------------------------------------------------------------

def test_ingest_documents_passes_lancedb_string_not_instance(monkeypatch, tmp_path):
    """GraphIngestor VdbUploadParams.vdb_op must be the backend id string.

    Passing a LanceDB instance as vdb_op gets stringified and fails
    get_vdb_op_cls with ``Invalid vdb_op: <LanceDB object ...>``.
    """
    calls = {}

    create_calls = {}

    class _FakeIngestor:
        def files(self, files):
            calls["files"] = files
            return self

        def extract(self, **kw):
            return self

        def embed(self, **kw):
            return self

        def vdb_upload(self, params=None, **kwargs):
            calls["vdb_upload_params"] = params
            calls["vdb_upload_kwargs"] = kwargs
            return self

        def ingest(self):
            calls["ingest"] = True

    def _fake_create_ingestor(**kw):
        create_calls.update(kw)
        return _FakeIngestor()

    import sys
    import types

    fake_nr = types.ModuleType("nemo_retriever")
    fake_nr.create_ingestor = _fake_create_ingestor
    fake_params = types.ModuleType("nemo_retriever.params")

    class _FakeEmbedParams:
        @classmethod
        def model_validate(cls, data):
            return data

    fake_params.EmbedParams = _FakeEmbedParams
    monkeypatch.setitem(sys.modules, "nemo_retriever", fake_nr)
    monkeypatch.setitem(sys.modules, "nemo_retriever.params", fake_params)

    pdf = tmp_path / "a.pdf"
    pdf.write_text("x", encoding="utf-8")
    uri = tmp_path / "idx"
    nrt._ingest_documents([str(pdf)], uri, "docs_test", {"hybrid": True})

    assert create_calls.get("run_mode") == "inprocess"
    assert calls.get("ingest") is True
    kw = calls.get("vdb_upload_kwargs") or {}
    assert kw.get("vdb_op") == "lancedb"
    assert isinstance(kw.get("vdb_op"), str)
    assert kw.get("vdb_kwargs", {}).get("table_name") == "docs_test"
    assert kw.get("vdb_kwargs", {}).get("uri") == str(uri)
    # Pinned SDK cannot hybrid-query; ingest follows the same dense-only flag.
    assert kw.get("vdb_kwargs", {}).get("hybrid") is False


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------

def test_registry_entry():
    from tools.registry import registry

    entry = registry.get_entry("document_search")
    assert entry is not None
    assert entry.toolset == "document_search"
    assert entry.check_fn is not None
    assert entry.check_fn.__name__ == "check_document_search_requirements"
    assert "NVIDIA_API_KEY" in entry.requires_env
    assert entry.emoji == "📚"


def test_toolset_registered():
    from toolsets import TOOLSETS, resolve_toolset

    assert "document_search" in TOOLSETS
    assert resolve_toolset("document_search") == ["document_search"]


def test_document_search_not_in_core_bundle():
    """Opt-in only — must NOT ship in the default core bundle (narrow waist)."""
    from toolsets import _HERMES_CORE_TOOLS

    assert "document_search" not in _HERMES_CORE_TOOLS
