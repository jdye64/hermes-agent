---
title: Document Search (NeMo Retriever)
description: Search PDF, Word, and HTML document collections by meaning and keyword using NVIDIA NeMo Retriever's BM25 + hybrid retrieval, instead of a ripgrep scan of binary files.
sidebar_label: Document Search
sidebar_position: 8
---

# Document Search (NVIDIA NeMo Retriever)

The `document_search` tool lets the agent search a collection of **documents** —
PDFs, Word files (`.doc` / `.docx`), and HTML pages — by keyword and meaning.
It is backed by [NVIDIA NeMo Retriever](https://github.com/NVIDIA/NeMo-Retriever):
documents are parsed and indexed into a local
[LanceDB](https://lancedb.github.io/lancedb/) table, and queries run through
**dense vector retrieval** (hybrid BM25 + dense is planned once the pinned
SDK supports it). The agent gets back the most relevant passages with their
source file and page.

**Use this instead of file search when the target is a document corpus.** A PDF
or `.docx` is a compressed binary — a ripgrep-style scan of the raw bytes
returns noise, not the text. `document_search` reads the *content*.
For source code and plain-text files, keep using regular file search.

## When it's available

`document_search` is an **opt-in, service-gated** tool. It stays completely out
of the model's tool schema — and the heavy `nemo-retriever` SDK stays
uninstalled — unless **both** of these are true:

1. The `document_search` toolset is enabled in `hermes tools`.
2. NeMo Retriever is configured, i.e. either:
   - `NVIDIA_API_KEY` is set (remote inference against
     [build.nvidia.com](https://build.nvidia.com/) or a NIM endpoint), **or**
   - `document_search.local: true` is set in `config.yaml` (local HuggingFace /
     vLLM embeddings; default).

```bash
hermes tools
# → 📚 Document Search   (press space to toggle on)
```

The first query over a new set of documents builds a local index (this can take
a while for large corpora and needs enough chunks to train LanceDB's index —
point it at a directory of documents, not a single tiny file). Later queries
reuse the index unless you pass `reindex: true`.

:::note Python 3.12
The `nemo-retriever` package requires Python 3.12. It is installed on demand
the first time the tool runs (like the Exa / Firecrawl web-search backends), and
is never part of the base install. On other interpreters the install fails with
a clear message rather than affecting the rest of Hermes.
:::

## Configuration

```yaml
# ~/.hermes/config.yaml
document_search:
  backend: nemo_retriever   # only backend today
  local: true               # strict local GPU HuggingFace / vLLM embeddings (default)
  top_k: 5                  # passages returned per query (1-50)
  hybrid: false             # dense vector retrieval (hybrid BM25+dense not usable on nemo-retriever 26.5.0)
  rerank: false             # remote-only in NRL 26.5.0; forced off in local mode
  index_dir: ""             # LanceDB directory; blank = HERMES_HOME/nemo_retriever
  embedding_model: nvidia/llama-nemotron-embed-1b-v2
  embedding_endpoint: ""    # ignored in local mode; local:false enables remote NIM
  local_ingest_embed_backend: hf  # hf (HuggingFace transformers) or vllm
  local_hf_device: "cuda:0" # required CUDA device; no CPU/remote fallback
  local_hf_cache_dir: ""    # optional Hugging Face cache override
  extract_method: ""        # blank = pdfium when local; "nemotron_parse" for scanned/image PDFs
  max_chars: 20000          # cap on returned passage text length
```

With `local: true`, ingest and query load `embedding_model` from the Hugging
Face Hub (or your local HF cache) onto the configured CUDA device. It is a
strict, fail-closed mode: configured HTTP embedding endpoints are ignored,
CPU fallback is rejected, reranking is disabled, and PDF extraction remains
pdfium text-only so page-element / table / chart NIMs are never contacted.
If CUDA is unavailable, the tool reports that directly and makes no remote
request. To use remote embeddings instead:

```yaml
document_search:
  local: false
  embedding_endpoint: https://integrate.api.nvidia.com/v1/embeddings
  embedding_model: nvidia/llama-nemotron-embed-1b-v2
```

On the pinned `nemo-retriever==26.5.0` SDK, queries use **dense** vector retrieval.
Setting `hybrid: true` is currently coerced to dense — that SDK raises
`NotImplementedError` for hybrid search over precomputed query vectors.
Indexes persist under `HERMES_HOME/nemo_retriever` by default, so each
[profile](../../reference/profiles.md) gets its own.

## Tool parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string (required) | Natural-language or keyword query. |
| `paths` | string array (required) | Document files (`.pdf`/`.doc`/`.docx`/`.html`/`.htm`) or directories containing them. |
| `top_k` | integer | Max passages to return (default from config, 1–50). |
| `index_name` | string | Optional name to reuse a persistent index across turns. Omit to derive one from the document set automatically. |
| `reindex` | boolean | Rebuild the index even if one already exists for these documents. |

The tool returns JSON with `backend`, `retrieval` (`dense`, or `bm25+hybrid` when
the SDK supports it), `index`, `indexed_documents`, `reindexed`, `count`, and a
`hits` array of `{text, score, source, page}` objects.

## Example

> Search my reports folder for what the Q3 filing says about supply-chain risk.

The agent will:

1. Call `document_search` with `query="Q3 filing supply-chain risk"` and
   `paths=["~/reports"]`.
2. On first use, parse and index every PDF/Word/HTML document in that folder.
3. Return the top passages — each with its source filename and page — for the
   model to summarize and cite.

## Troubleshooting

### "Document search is not configured"

Set `document_search.local: true` in `config.yaml` (default) to use local
HuggingFace embedding models with no remote embedding endpoint, **or** set
`NVIDIA_API_KEY` in `~/.hermes/.env` (from
[build.nvidia.com](https://build.nvidia.com/)) and an `embedding_endpoint`
URL for remote NIM embeddings. Then restart your
session so the agent re-reads the tool registry.

### "strict local GPU inference" / `torch.cuda.is_available() is false`

Verify all three checks from the same environment that launches Hermes:

```bash
nvidia-smi
ls -l /dev/nvidia*
/path/to/hermes-agent/.venv/bin/python -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

On a DGX Spark after a kernel update, ensure the NVIDIA kernel module package
matches `uname -r`; CUDA user-space libraries alone are not sufficient.

### Tool doesn't appear in the schema

Two possible causes:

1. **Toolset not enabled.** Run `hermes tools` and confirm `📚 Document Search`
   is checked.
2. **Not configured.** The `check_fn` returns `False` when neither
   `NVIDIA_API_KEY` nor `document_search.local` is set, so the schema stays
   hidden.

### "NVIDIA NeMo Retriever SDK is not installed"

The lazy install failed — most often because the interpreter is not Python 3.12,
or lazy installs are disabled (`security.allow_lazy_installs: false`). Install
manually into the agent's environment with `pip install nemo-retriever==26.5.0`
under a Python 3.12 venv.

## See Also

- [Web Search & Extract](web-search.md) — for general web pages and PDF *URLs*
- [Tools Reference](../../reference/tools-reference.md) — full tool catalog
- [NVIDIA NeMo Retriever](https://github.com/NVIDIA/NeMo-Retriever) — upstream SDK
