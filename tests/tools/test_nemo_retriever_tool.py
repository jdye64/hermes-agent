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
    assert out["retrieval"] == "bm25+hybrid"
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


def test_bm25_only_when_hybrid_disabled(monkeypatch, tmp_path):
    _enable(monkeypatch, {"hybrid": False})
    pdf = tmp_path / "a.pdf"
    pdf.write_text("x", encoding="utf-8")
    monkeypatch.setattr(nrt, "_nemo_retriever_search", lambda **k: ([], False))
    out = json.loads(nrt.document_search(query="q", paths=[str(pdf)]))
    assert out["retrieval"] == "bm25"


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
