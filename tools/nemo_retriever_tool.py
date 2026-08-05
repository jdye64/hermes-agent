#!/usr/bin/env python3
"""Document search backed by NVIDIA NeMo Retriever (``document_search`` tool).

When a user asks to *search* a collection of **documents** — PDFs, Word files
(``.doc`` / ``.docx``), or HTML pages — the right primitive is retrieval over
the document *content*, not a ripgrep-style regex scan of the raw bytes (a PDF
is a compressed binary; grepping it returns noise). This tool routes that
"search my documents" intent through NVIDIA NeMo Retriever's BM25 lexical +
dense hybrid retrieval so the model gets back the most relevant passages with
their source file and page.

Footprint
---------
This is a **service-gated** tool (Footprint Ladder rung 3). It lives in its own
opt-in ``document_search`` toolset (NOT in ``_HERMES_CORE_TOOLS``) and its
``check_fn`` only reports available when a NeMo Retriever credential is present
(``NVIDIA_API_KEY`` for remote build.nvidia.com / NIM inference, or
``document_search.local: true`` for a local GPU deployment). With neither
configured the tool never reaches the model schema — zero permanent footprint.
Ripgrep-backed ``search_files`` stays the tool for code/text; this is
purely for document *corpora*.

The heavy ``nemo-retriever`` SDK (Python 3.12-only, LanceDB, embedding NIM
clients) is **never** a core dependency. It is imported lazily and, if missing,
installed on demand through :func:`tools.lazy_deps.ensure` — exactly like the
Exa / Firecrawl / Parallel web-search backends.

SDK reference: https://github.com/NVIDIA/NeMo-Retriever (the ``nemo-retriever``
PyPI package). Ingestion uses ``create_ingestor(...).files(...).extract()``
``.embed().vdb_upload(vdb_op="lancedb", vdb_kwargs=...)``; querying uses
``nemo_retriever.retriever.Retriever(...).query(...)``. BM25 is the LanceDB
full-text component of hybrid retrieval (BM25 FTS + dense, fused with RRF).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

# Documents the user means when they say "search my PDFs / Word docs / web
# pages". Deliberately narrow: code and plain-text search belong to the
# ripgrep-backed file search, not to a heavyweight retrieval index.
SUPPORTED_EXTENSIONS = frozenset({".pdf", ".doc", ".docx", ".html", ".htm"})

DEFAULT_EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
DEFAULT_EMBEDDING_ENDPOINT = "https://integrate.api.nvidia.com/v1/embeddings"
_LAZY_FEATURE = "document_search.nemo_retriever"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load the ``document_search:`` section from config.yaml.

    Mirrors :func:`tools.web_tools._load_web_config`: honor the ``-> dict``
    contract even when the section is present-but-null (YAML ``document_search:``
    with no body yields ``None`` from ``.get``).
    """
    try:
        from hermes_cli.config import load_config

        return load_config().get("document_search") or {}
    except Exception:
        return {}


def check_document_search_requirements() -> bool:
    """``check_fn`` gate: available only when NeMo Retriever is configured.

    True when a remote NeMo Retriever credential is set (``NVIDIA_API_KEY``)
    or the user has opted into a local GPU deployment
    (``document_search.local: true``). This keeps the tool — and the entire
    heavy ``nemo-retriever`` dependency — invisible to every session that
    hasn't configured it.
    """
    try:
        if os.getenv("NVIDIA_API_KEY", "").strip():
            return True
        return bool(_load_config().get("local"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _collect_documents(paths: List[str]) -> tuple[List[str], List[str]]:
    """Expand ``paths`` into concrete supported document files.

    Directories are walked (non-recursively-hidden) for supported extensions;
    individual files are kept when their extension is supported. Returns
    ``(files, skipped)`` where ``skipped`` records inputs that were ignored so
    the caller can report them.
    """
    files: List[str] = []
    skipped: List[str] = []
    seen: set[str] = set()

    for raw in paths:
        candidate = Path(os.path.expanduser(str(raw))).resolve()
        if candidate.is_dir():
            matched_any = False
            for child in sorted(candidate.rglob("*")):
                if not child.is_file():
                    continue
                # Skip dotfiles / dot-directories (e.g. .git, .venv).
                if any(part.startswith(".") for part in child.relative_to(candidate).parts):
                    continue
                if child.suffix.lower() in SUPPORTED_EXTENSIONS:
                    key = str(child)
                    if key not in seen:
                        seen.add(key)
                        files.append(key)
                        matched_any = True
            if not matched_any:
                skipped.append(f"{candidate} (no PDF/Word/HTML documents found)")
        elif candidate.is_file():
            if candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                key = str(candidate)
                if key not in seen:
                    seen.add(key)
                    files.append(key)
            else:
                skipped.append(f"{candidate} (unsupported type {candidate.suffix or '<none>'})")
        else:
            skipped.append(f"{raw} (not found)")

    return files, skipped


def _index_root() -> Path:
    """Base directory for persisted LanceDB indexes (profile-aware)."""
    cfg_dir = str(_load_config().get("index_dir") or "").strip()
    if cfg_dir:
        return Path(os.path.expanduser(cfg_dir))
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "nemo_retriever"


def _derive_table_name(files: List[str], index_name: Optional[str]) -> str:
    """Stable LanceDB table name for a document set.

    An explicit ``index_name`` wins so callers can reuse a named corpus across
    turns. Otherwise the name is derived from the sorted absolute file paths so
    the same set of documents maps to the same index (and skips re-ingestion).
    """
    if index_name and index_name.strip():
        cleaned = "".join(c if (c.isalnum() or c in "-_") else "_" for c in index_name.strip())
        return f"docs_{cleaned}"[:120]
    digest = hashlib.sha256("\n".join(sorted(files)).encode("utf-8")).hexdigest()[:16]
    return f"docs_{digest}"


def _table_exists(uri: Path, table_name: str) -> bool:
    """Whether a LanceDB table already exists under ``uri``.

    LanceDB persists each table as ``<uri>/<table_name>.lance``.
    """
    return (uri / f"{table_name}.lance").exists()


# ---------------------------------------------------------------------------
# NeMo Retriever SDK boundary (isolated for testability)
# ---------------------------------------------------------------------------

def _ensure_sdk() -> None:
    """Lazily install the ``nemo-retriever`` SDK, raising a clear error.

    Kept separate from the SDK imports so unit tests can monkeypatch the
    higher-level search boundary without touching pip.
    """
    from tools.lazy_deps import FeatureUnavailable, ensure

    try:
        ensure(_LAZY_FEATURE, prompt=False)
    except FeatureUnavailable as exc:
        raise RuntimeError(
            "NVIDIA NeMo Retriever SDK is not installed. "
            f"{exc} The nemo-retriever package requires Python 3.12."
        ) from exc


def _ingest_documents(files: List[str], uri: Path, table_name: str, cfg: dict) -> None:
    """Ingest ``files`` into a LanceDB table with BM25/hybrid retrieval enabled.

    GraphIngestor's ``VdbUploadParams.vdb_op`` is a *string* backend id
    (``"lancedb"``), not a ``LanceDB`` instance — passing an instance gets
    stringified and fails ``get_vdb_op_cls`` with ``Invalid vdb_op: <LanceDB...>``.
    Table location goes in ``vdb_kwargs``.
    """
    from nemo_retriever import create_ingestor  # type: ignore

    uri.mkdir(parents=True, exist_ok=True)
    # FTS index is only useful if query-time hybrid works. On the pinned
    # nemo-retriever 26.5.0 it does not (see ``_query_index``), so default
    # ingest to dense-only unless/until the SDK supports hybrid query.
    hybrid = _query_hybrid_enabled(cfg)

    ingestor = create_ingestor(run_mode="inprocess")
    ingestor = ingestor.files(files)

    extract_method = str(cfg.get("extract_method") or "").strip()
    if extract_method:
        ingestor = ingestor.extract(method=extract_method)
    else:
        ingestor = ingestor.extract()

    ingestor = ingestor.embed(
        model_name=str(cfg.get("embedding_model") or DEFAULT_EMBEDDING_MODEL),
        embed_invoke_url=str(cfg.get("embedding_endpoint") or DEFAULT_EMBEDDING_ENDPOINT),
        embed_modality="text",
    )
    # vdb_op must be the backend name string; kwargs configure LanceDB.
    ingestor = ingestor.vdb_upload(
        vdb_op="lancedb",
        vdb_kwargs={
            "uri": str(uri),
            "table_name": table_name,
            "hybrid": hybrid,
        },
    )
    ingestor.ingest()


def _query_hybrid_enabled(cfg: dict) -> bool:
    """Whether to request LanceDB hybrid (BM25 + dense) at query time.

    ``nemo-retriever==26.5.0`` (our lazy pin) embeds the query then calls
    ``LanceDB.retrieval(vectors, ...)``. That path raises
    ``NotImplementedError: LanceDB hybrid retrieval with precomputed vectors
    is not implemented yet`` when ``hybrid=True``. Newer SDK mainline adds
    hybrid+``query_texts`` support; until we bump the pin, force dense-only
    so ``document_search`` works. Respect ``hybrid: false`` explicitly;
    ``hybrid: true`` is accepted but currently coerced to dense.
    """
    # Keep the config knob, but do not enable hybrid against the broken pin.
    if not bool(cfg.get("hybrid", False)):
        return False
    return False  # flip when lazy_deps pin gains working hybrid query


def _query_index(query: str, uri: Path, table_name: str, top_k: int, cfg: dict) -> List[dict]:
    """Run dense (or hybrid, when supported) retrieval against an existing LanceDB table."""
    from nemo_retriever.retriever import Retriever  # type: ignore

    hybrid = _query_hybrid_enabled(cfg)
    vdb_kwargs: Dict[str, Any] = {
        "uri": str(uri),
        "table_name": table_name,
        "hybrid": hybrid,
    }
    embed_model = str(cfg.get("embedding_model") or DEFAULT_EMBEDDING_MODEL)
    retriever = Retriever(
        vdb_kwargs=vdb_kwargs,
        embed_kwargs={
            "model_name": embed_model,
            "embed_model_name": embed_model,
            "embedding_endpoint": str(cfg.get("embedding_endpoint") or DEFAULT_EMBEDDING_ENDPOINT),
        },
        top_k=top_k,
        rerank=bool(cfg.get("rerank", False)),
    )
    try:
        hits = retriever.query(query)
    except NotImplementedError as exc:
        # Defensive: if a future pin partially enables hybrid but still hits
        # this path, fall back to dense rather than failing the tool.
        if not hybrid:
            raise
        logger.warning(
            "NeMo Retriever hybrid query unsupported (%s); retrying with dense retrieval",
            exc,
        )
        vdb_kwargs = {**vdb_kwargs, "hybrid": False}
        retriever = Retriever(
            vdb_kwargs=vdb_kwargs,
            embed_kwargs={
                "model_name": embed_model,
                "embed_model_name": embed_model,
                "embedding_endpoint": str(
                    cfg.get("embedding_endpoint") or DEFAULT_EMBEDDING_ENDPOINT
                ),
            },
            top_k=top_k,
            rerank=bool(cfg.get("rerank", False)),
        )
        hits = retriever.query(query)
    return list(hits or [])


def _nemo_retriever_search(
    query: str,
    files: List[str],
    uri: Path,
    table_name: str,
    top_k: int,
    reindex: bool,
    cfg: dict,
) -> tuple[List[dict], bool]:
    """Orchestrate ingest-if-needed + query. Returns ``(hits, did_ingest)``.

    This is the single SDK-touching entry point; tests monkeypatch it (or the
    two helpers it calls) so the suite never needs the real ``nemo-retriever``
    package or a GPU/API key.
    """
    _ensure_sdk()
    did_ingest = False
    if reindex or not _table_exists(uri, table_name):
        _ingest_documents(files, uri, table_name, cfg)
        did_ingest = True
    hits = _query_index(query, uri, table_name, top_k, cfg)
    return hits, did_ingest


# ---------------------------------------------------------------------------
# Hit normalization
# ---------------------------------------------------------------------------

def _normalize_hit(hit: Any, max_chars: int) -> dict:
    """Normalize an SDK hit into a stable ``{text, score, source, page}`` shape.

    NeMo Retriever hit dicts always carry ``text``; score and source metadata
    keys vary by version/backend, so probe the common aliases defensively.
    """
    if not isinstance(hit, dict):
        return {"text": str(hit)[:max_chars], "score": None, "source": None, "page": None}

    text = str(hit.get("text") or hit.get("content") or "")
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"

    score = hit.get("score")
    if score is None:
        score = hit.get("_relevance_score", hit.get("_distance", hit.get("rank")))

    meta = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    source_meta = meta.get("source_metadata") if isinstance(meta.get("source_metadata"), dict) else {}
    source = (
        hit.get("source")
        or hit.get("source_name")
        or meta.get("source")
        or meta.get("source_name")
        or source_meta.get("source_name")
        or source_meta.get("source_id")
    )
    page = hit.get("page") or hit.get("page_number") or meta.get("page") or meta.get("page_number")

    return {"text": text, "score": score, "source": source, "page": page}


# ---------------------------------------------------------------------------
# Public tool
# ---------------------------------------------------------------------------

def document_search(
    query: str,
    paths: Optional[List[str]] = None,
    top_k: Optional[int] = None,
    reindex: bool = False,
    index_name: Optional[str] = None,
) -> str:
    """Search PDF/Word/HTML documents via NeMo Retriever BM25/hybrid retrieval."""
    query = (query or "").strip()
    if not query:
        return tool_error("`query` is required and must be a non-empty string.")

    if not paths:
        return tool_error(
            "`paths` is required: pass one or more document files or a "
            "directory containing PDF/Word/HTML documents to search."
        )
    if isinstance(paths, str):
        paths = [paths]

    if not check_document_search_requirements():
        return tool_error(
            "Document search is not configured. Set NVIDIA_API_KEY (from "
            "https://build.nvidia.com/) for remote NeMo Retriever inference, or "
            "set `document_search.local: true` in config.yaml for a local GPU "
            "deployment."
        )

    cfg = _load_config()
    files, skipped = _collect_documents([str(p) for p in paths])
    if not files:
        return tool_error(
            "No PDF, Word (.doc/.docx), or HTML documents found in the given "
            "paths. Use file search for code/plain-text.",
            skipped=skipped,
        )

    resolved_top_k = top_k if isinstance(top_k, int) and top_k > 0 else int(cfg.get("top_k", 5) or 5)
    resolved_top_k = max(1, min(resolved_top_k, 50))
    max_chars = int(cfg.get("max_chars", 20000) or 20000)

    uri = _index_root()
    table_name = _derive_table_name(files, index_name)

    try:
        raw_hits, did_ingest = _nemo_retriever_search(
            query=query,
            files=files,
            uri=uri,
            table_name=table_name,
            top_k=resolved_top_k,
            reindex=bool(reindex),
            cfg=cfg,
        )
    except RuntimeError as exc:
        return tool_error(str(exc))
    except Exception as exc:  # SDK / network / ingestion failures
        logger.warning("document_search failed: %s", exc, exc_info=True)
        return tool_error(f"NeMo Retriever document search failed: {exc}")

    hits = [_normalize_hit(h, max_chars) for h in raw_hits[:resolved_top_k]]

    result: Dict[str, Any] = {
        "success": True,
        "backend": "nemo_retriever",
        "retrieval": "bm25+hybrid" if _query_hybrid_enabled(cfg) else "dense",
        "query": query,
        "index": table_name,
        "indexed_documents": len(files),
        "reindexed": did_ingest,
        "count": len(hits),
        "hits": hits,
    }
    if skipped:
        result["skipped"] = skipped
    return json.dumps(result, ensure_ascii=False)


DOCUMENT_SEARCH_SCHEMA = {
    "name": "document_search",
    "description": (
        "Search a collection of DOCUMENTS — PDFs, Word files (.doc/.docx), and "
        "HTML pages — by keyword and meaning using NVIDIA NeMo Retriever (BM25 "
        "lexical + dense hybrid retrieval). Use this whenever the user asks to "
        "'search' such documents (reports, papers, contracts, manuals, saved "
        "web pages): it parses the document content and returns the most "
        "relevant passages with their source file and page, which a raw text "
        "scan of a binary PDF/Word file cannot do. Point it at individual "
        "document files or a directory of them. The first query over a new set "
        "of documents builds a local index (this can take a while for large "
        "corpora); later queries reuse it unless reindex=true. NOTE: this is "
        "for document corpora only — for source code or plain-text files, use "
        "regular file search instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language or keyword query to search the documents for.",
            },
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Document files (.pdf/.doc/.docx/.html/.htm) or directories "
                    "containing them to search."
                ),
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of passages to return (default from config, 1-50).",
                "minimum": 1,
                "maximum": 50,
            },
            "index_name": {
                "type": "string",
                "description": (
                    "Optional name to reuse a persistent index across turns. "
                    "Omit to derive one from the document set automatically."
                ),
            },
            "reindex": {
                "type": "boolean",
                "description": "Rebuild the index even if one already exists for these documents.",
                "default": False,
            },
        },
        "required": ["query", "paths"],
    },
}


def _handle_document_search(args, **kw):
    return document_search(
        query=args.get("query", ""),
        paths=args.get("paths"),
        top_k=args.get("top_k"),
        reindex=bool(args.get("reindex", False)),
        index_name=args.get("index_name"),
    )


registry.register(
    name="document_search",
    toolset="document_search",
    schema=DOCUMENT_SEARCH_SCHEMA,
    handler=_handle_document_search,
    check_fn=check_document_search_requirements,
    requires_env=["NVIDIA_API_KEY"],
    emoji="📚",
    max_result_size_chars=100_000,
)
