#!/usr/bin/env python3
"""Benchmark the Hermes ``document_search`` tool (NVIDIA NeMo Retriever) over a
PDF corpus, driving it through the Hermes agent CLI in one-shot mode.

Every query comes from a ground-truth CSV (``query`` column); the CSV's
``dir``/``pdf``/``page`` columns are used to score whether retrieval actually
found the right page.

Reports **pages per second** and **total wall time** for the run.

Two modes, so you get a baseline with and without the NRL tool:

    --mode document_search   WITH NRL   — the agent calls ``document_search``
    --mode baseline          WITHOUT NRL — same queries, only file/terminal tools
    --mode both              run each in turn, then print a side-by-side compare

Usage:
    # both modes, saving every artifact under ./benchmark_results
    python scripts/benchmark_document_search.py --mode both --out-dir benchmark_results

    # one mode at a time
    python scripts/benchmark_document_search.py --mode document_search --out-dir benchmark_results
    python scripts/benchmark_document_search.py --mode baseline        --out-dir benchmark_results

    # smoke test / inspect the exact CLI command without running it
    python scripts/benchmark_document_search.py --mode both --limit 5 --out-dir /tmp/smoke
    python scripts/benchmark_document_search.py --mode both --dry-run

With ``--out-dir DIR`` each mode writes:
    DIR/<mode>.jsonl          per-query result rows (hits, timings, hit/miss)
    DIR/<mode>.summary.txt    the human-readable report
    DIR/<mode>.summary.json   the same metrics, machine-readable
and ``--mode both`` additionally writes DIR/comparison.txt and DIR/comparison.json.

Requirements for the WITH-NRL mode (same gate as the tool itself):
    - ``NVIDIA_API_KEY`` set, or ``document_search.local: true`` in config.yaml, and
    - a Python 3.12 venv — ``nemo-retriever==26.5.0`` requires >=3.12,<3.13.
The baseline mode needs neither.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

DEFAULT_CORPUS = "/home/jdyer/datasets/earnings_v2/earnings_consulting"
DEFAULT_QUERIES = "/home/jdyer/datasets/earnings_v2/earnings_consulting_multimodal_v2.csv"
DOC_SUFFIXES = {".pdf", ".doc", ".docx", ".html", ".htm"}

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Corpus / query set
# ---------------------------------------------------------------------------

def count_pages(corpus: Path) -> tuple[int, int, str]:
    """Return (total_pages, n_documents, method) for the corpus.

    Tries pypdf, then ``pdfinfo``. Non-PDF documents count as 1 page.
    """
    docs = [p for p in sorted(corpus.rglob("*")) if p.suffix.lower() in DOC_SUFFIXES]
    pdfs = [p for p in docs if p.suffix.lower() == ".pdf"]
    other = len(docs) - len(pdfs)

    total = other
    method = "none"

    try:
        from pypdf import PdfReader  # type: ignore

        method = "pypdf"
        for p in pdfs:
            try:
                total += len(PdfReader(str(p), strict=False).pages)
            except Exception as exc:
                log(f"  ! page count failed for {p.name}: {exc} (counted as 1)")
                total += 1
        return total, len(docs), method
    except ImportError:
        pass

    if shutil.which("pdfinfo"):
        method = "pdfinfo"
        for p in pdfs:
            try:
                out = subprocess.run(
                    ["pdfinfo", str(p)], capture_output=True, text=True, timeout=60
                ).stdout
                pages = next(
                    (int(l.split(":", 1)[1]) for l in out.splitlines() if l.startswith("Pages:")),
                    1,
                )
                total += pages
            except Exception:
                total += 1
        return total, len(docs), method

    log("  ! neither pypdf nor pdfinfo available — falling back to CSV max page per doc")
    return 0, len(docs), "unavailable"


def pages_from_csv(rows: list[dict]) -> int:
    """Lower-bound page count: the highest ground-truth page seen per document."""
    best: dict[tuple[str, str], int] = {}
    for r in rows:
        key = (r.get("dir", ""), r.get("pdf", ""))
        try:
            page = int(r.get("page") or 0)
        except ValueError:
            page = 0
        best[key] = max(best.get(key, 0), page)
    return sum(best.values())


def load_queries(path: Path, limit: Optional[int]) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("query") or "").strip()]
    return rows[:limit] if limit else rows


# ---------------------------------------------------------------------------
# Hermes CLI invocation
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    idx: int
    query: str
    seconds: float
    ok: bool
    hits: list[dict] = field(default_factory=list)
    error: str = ""
    raw: str = ""
    answer: str = ""
    doc_hit: bool = False
    page_hit: bool = False
    answer_match: bool = False


def build_prompt(args, query: str, reindex: bool = False) -> str:
    if args.mode == "baseline":
        return build_prompt_baseline(query, args.corpus, args.top_k)
    return build_prompt_ds(query, args.corpus, args.index_name, args.top_k, reindex)


def build_prompt_ds(query: str, corpus: str, index_name: str, top_k: int, reindex: bool) -> str:
    return (
        "Call the document_search tool exactly once, then stop.\n"
        f'  query: "{query}"\n'
        f'  paths: ["{corpus}"]\n'
        f'  index_name: "{index_name}"\n'
        f"  top_k: {top_k}\n"
        f"  reindex: {'true' if reindex else 'false'}\n"
        "Do not call any other tool and do not search the filesystem.\n"
        "Reply with ONLY the raw JSON object the tool returned — no commentary, "
        "no markdown fences, no summary."
    )


def build_prompt_baseline(query: str, corpus: str, top_k: int) -> str:
    """No document_search: the agent must locate the answer with ordinary tools."""
    return (
        f"The directory {corpus} holds a corpus of PDF documents in subfolders.\n"
        "Using only those documents, answer this question:\n"
        f'  "{query}"\n'
        "Locate the specific document and page that contains the answer.\n"
        "When you have it, reply with ONLY this JSON object and nothing else — "
        "no commentary, no markdown fences:\n"
        '  {"answer": "<the answer>", "source": "<pdf file name>", "page": <page number>}\n'
        'If you cannot find it, reply with {"answer": null, "source": null, "page": null}.'
    )


def build_cmd(args, prompt: str) -> list[str]:
    cmd = [args.hermes, "-z", prompt, "-t", args.toolsets]
    if args.model:
        cmd += ["--model", args.model]
    if args.provider:
        cmd += ["--provider", args.provider]
    if args.reasoning:
        cmd += ["--reasoning", args.reasoning]
    return cmd


def extract_json(text: str) -> Optional[dict]:
    """Pull the document_search result object out of the agent's stdout.

    Only *top-level* objects are considered. Advancing the cursor past each
    decoded object matters: every ``document_search`` hit is itself a dict
    carrying a ``source`` key, so a naive per-character scan would let the
    last nested hit shadow the real payload and report zero hits.

    Among top-level candidates, an object that actually looks like a tool
    result (``hits``/``error``) wins over a weaker ``answer``/``source``
    match; otherwise the last candidate wins, since the agent may echo the
    prompt before printing the result.
    """
    decoder = json.JSONDecoder()
    strong: Optional[dict] = None
    weak: Optional[dict] = None
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(text[i:])
        except ValueError:
            i += 1
            continue
        i += end  # skip the whole object; never descend into nested dicts
        if not isinstance(obj, dict):
            continue
        if "hits" in obj or "error" in obj:
            strong = obj
        elif "answer" in obj or "source" in obj:
            weak = obj
    return strong if strong is not None else weak


def run_query(args, row: dict, idx: int, reindex: bool = False) -> QueryResult:
    query = row["query"].strip()
    prompt = build_prompt(args, query, reindex)
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            build_cmd(args, prompt),
            capture_output=True,
            text=True,
            timeout=args.timeout,
            cwd=args.cwd,
        )
        out, err, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return QueryResult(idx, query, time.perf_counter() - started, False,
                           error=f"timeout after {args.timeout}s")
    elapsed = time.perf_counter() - started

    payload = extract_json(out)
    if payload is None:
        tail = (err or out or "").strip().splitlines()[-3:]
        return QueryResult(idx, query, elapsed, False, raw=out,
                           error=f"rc={rc}; no tool JSON in output: {' | '.join(tail)[:300]}")
    if payload.get("error") or payload.get("success") is False:
        return QueryResult(idx, query, elapsed, False, raw=out,
                           error=str(payload.get("error"))[:300])

    if args.mode == "baseline":
        # single {answer, source, page} object -> a one-element hit list
        answer = payload.get("answer")
        hits = [] if payload.get("source") in (None, "") else [
            {"source": payload.get("source"), "page": payload.get("page"), "score": None}
        ]
        res = QueryResult(idx, query, elapsed, True, hits=hits, raw=out,
                          answer="" if answer is None else str(answer))
    else:
        hits = [h for h in payload.get("hits", []) if isinstance(h, dict)]
        res = QueryResult(idx, query, elapsed, True, hits=hits, raw=out)
    score_hit(res, row)
    return res


def score_hit(res: QueryResult, row: dict) -> None:
    """Mark whether the ground-truth document / page appears in the top-k hits."""
    want_doc = (row.get("pdf") or "").strip().lower()
    try:
        want_page = int(row.get("page") or 0)
    except ValueError:
        want_page = 0

    for h in res.hits:
        source = str(h.get("source") or "")
        stem = Path(source).stem.lower()
        if want_doc and (want_doc == stem or want_doc in source.lower()):
            res.doc_hit = True
            page = h.get("page")
            try:
                page = int(page)
            except (TypeError, ValueError):
                page = None
            # tolerate 0- vs 1-indexed page numbering
            if page is not None and want_page and abs(page - want_page) <= 1:
                res.page_hit = True
    if not want_page:
        res.page_hit = res.doc_hit

    gold = (row.get("answer") or "").strip().lower()
    got = (res.answer or "").strip().lower()
    if gold and got:
        res.answer_match = gold in got or got in gold


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def _preflight_paths(args, problems: list) -> None:
    if not Path(args.corpus).is_dir():
        problems.append(f"corpus directory not found: {args.corpus}")
    if not Path(args.queries).is_file():
        problems.append(f"queries CSV not found: {args.queries}")
    if not (Path(args.hermes).exists() or shutil.which(args.hermes)):
        problems.append(f"hermes CLI not found: {args.hermes} (pass --hermes PATH)")


def preflight(args) -> None:
    problems = []
    if args.mode == "baseline":
        _preflight_paths(args, problems)
        if problems:
            for p in problems:
                print(f"error: {p}", file=sys.stderr)
            sys.exit(2)
        return

    _preflight_paths(args, problems)

    configured = bool(os.environ.get("NVIDIA_API_KEY"))
    if not configured:
        cfg_path = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / "config.yaml"
        try:
            import yaml  # type: ignore

            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            ds = cfg.get("document_search") or {}
            configured = bool(ds.get("local")) or bool(
                (cfg.get("env") or {}).get("NVIDIA_API_KEY")
            )
        except Exception:
            pass
    if not configured:
        log(
            "  ! NVIDIA_API_KEY is not set and document_search.local is not true.\n"
            "    document_search will stay out of the tool schema and every query "
            "will fail.\n"
            "    Fix: export NVIDIA_API_KEY=... (or set document_search.local: true "
            "in ~/.hermes/config.yaml)."
        )

    if problems:
        for p in problems:
            print(f"error: {p}", file=sys.stderr)
        sys.exit(2)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt_hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else (f"{m:d}m{s:02d}s" if m else f"{s:d}s")


MODE_LABEL = {
    "document_search": "WITH NRL (NeMo Retriever document_search)",
    "baseline": "WITHOUT NRL (baseline — ordinary file/terminal tools)",
}


def metrics(args, results: list[QueryResult], total_pages: int, page_method: str,
            n_docs: int, index_seconds: float, wall: float) -> dict:
    """Machine-readable summary of one mode's run."""
    ok = [r for r in results if r.ok]
    lat = sorted(r.seconds for r in ok)

    m: dict[str, Any] = {
        "mode": args.mode,
        "label": MODE_LABEL[args.mode],
        "toolsets": args.toolsets,
        "corpus": args.corpus,
        "documents": n_docs,
        "pages": total_pages,
        "page_count_method": page_method,
        "queries": len(results),
        "queries_ok": len(ok),
        "queries_failed": len(results) - len(ok),
        "concurrency": args.concurrency,
        "top_k": args.top_k,
        "model": args.model,
        "provider": args.provider,
        # ---- headline numbers ----
        "total_wall_seconds": round(wall, 3),
        "total_wall_hms": fmt_hms(wall),
        "pages_per_second": round(total_pages / wall, 4) if (total_pages and wall > 0) else None,
        "effective_scan_pages_per_second": (
            round(total_pages * len(results) / wall, 4) if (total_pages and wall > 0) else None
        ),
        "queries_per_second": round(len(results) / wall, 4) if wall > 0 else None,
        "index_seconds": round(index_seconds, 3),
        "ingest_pages_per_second": (
            round(total_pages / index_seconds, 4) if (total_pages and index_seconds > 0) else None
        ),
    }
    if lat:
        m["latency_seconds"] = {
            "mean": round(statistics.mean(lat), 3),
            "median": round(statistics.median(lat), 3),
            "p95": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 3),
            "max": round(lat[-1], 3),
        }
    if ok:
        m["doc_hit_rate"] = round(sum(r.doc_hit for r in ok) / len(ok), 4)
        m["page_hit_rate"] = round(sum(r.page_hit for r in ok) / len(ok), 4)
        m["answer_match_rate"] = round(sum(r.answer_match for r in ok) / len(ok), 4)
    m["failures"] = [
        {"idx": r.idx, "query": r.query[:120], "error": r.error[:300]}
        for r in results if not r.ok
    ][:20]
    return m


def report(args, results: list[QueryResult], total_pages: int, page_method: str,
           n_docs: int, index_seconds: float, wall: float) -> str:
    """Render the human-readable report and return it as text."""
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    lat = sorted(r.seconds for r in ok)
    out: list[str] = []
    p = out.append

    p("")
    p("=" * 68)
    p(f"  Hermes benchmark — {MODE_LABEL[args.mode]}")
    p("=" * 68)
    p(f"  corpus            : {args.corpus}")
    p(f"  documents         : {n_docs}")
    p(f"  pages             : {total_pages}  (counted via {page_method})")
    p(f"  queries           : {len(results)}  ({len(ok)} ok, {len(failed)} failed)")
    p(f"  toolsets          : {args.toolsets}")
    p(f"  concurrency       : {args.concurrency}   top_k: {args.top_k}")
    p("-" * 68)
    if index_seconds:
        p(f"  index/ingest time : {index_seconds:.2f}s"
          + (f"   ({total_pages / index_seconds:.2f} pages/s ingest)"
             if total_pages and index_seconds > 0 else ""))
        p("")
    p(f"  TOTAL WALL TIME   : {wall:.2f} s  ({fmt_hms(wall)})")
    if total_pages and wall > 0:
        # Headline throughput: the corpus is indexed once and every query runs
        # against that one index, so the run processes `total_pages` pages of
        # source material end to end.
        p(f"  PAGES PER SECOND  : {total_pages / wall:.2f} pages/s"
          f"   ({total_pages} pages / {wall:.2f}s)")
        p(f"  effective scan rate: {total_pages * len(results) / wall:.2f} pages/s"
          f"   ({total_pages} pages x {len(results)} queries / {wall:.2f}s)"
          "  [pages a linear scan would have had to read]")
    else:
        p("  PAGES PER SECOND  : n/a (page count unavailable)")
    if wall > 0:
        p(f"  queries per second: {len(results) / wall:.3f} q/s")
    if lat:
        p(f"  query latency     : mean {statistics.mean(lat):.2f}s | "
          f"median {statistics.median(lat):.2f}s | "
          f"p95 {lat[min(len(lat) - 1, int(0.95 * len(lat)))]:.2f}s | "
          f"max {lat[-1]:.2f}s")
    if ok:
        p("-" * 68)
        p(f"  retrieval doc@{args.top_k}     : "
          f"{sum(r.doc_hit for r in ok) / len(ok) * 100:.1f}%  "
          f"({sum(r.doc_hit for r in ok)}/{len(ok)})")
        p(f"  retrieval page@{args.top_k}    : "
          f"{sum(r.page_hit for r in ok) / len(ok) * 100:.1f}%  "
          f"({sum(r.page_hit for r in ok)}/{len(ok)})")
        if args.mode == "baseline":
            p(f"  answer match      : "
              f"{sum(r.answer_match for r in ok) / len(ok) * 100:.1f}%  "
              f"({sum(r.answer_match for r in ok)}/{len(ok)})"
              "   [substring vs ground truth]")
    if failed:
        p("-" * 68)
        p(f"  first failures ({len(failed)} total):")
        for r in failed[:5]:
            p(f"    [{r.idx}] {r.query[:52]!r} -> {r.error[:120]}")
    p("=" * 68)

    text = "\n".join(out)
    print(text)
    return text


def compare_report(summaries: list[dict]) -> str:
    """Side-by-side of the WITH-NRL and WITHOUT-NRL runs."""
    by_mode = {s["mode"]: s for s in summaries}
    ds, base = by_mode.get("document_search"), by_mode.get("baseline")
    out: list[str] = []
    p = out.append

    p("")
    p("=" * 68)
    p("  COMPARISON — NRL (NeMo Retriever) vs baseline")
    p("=" * 68)
    p(f"  {'metric':<26}{'WITH NRL':>19}{'WITHOUT NRL':>19}")
    p("-" * 68)

    def fmt(v, suffix="", nd=2):
        if v is None:
            return "n/a"
        return f"{v:.{nd}f}{suffix}" if isinstance(v, float) else f"{v}{suffix}"

    rows = [
        ("total wall time (s)", "total_wall_seconds", "", 2),
        ("PAGES PER SECOND", "pages_per_second", "", 2),
        ("queries per second", "queries_per_second", "", 3),
        ("index/ingest time (s)", "index_seconds", "", 2),
        ("queries ok", "queries_ok", "", 0),
        ("queries failed", "queries_failed", "", 0),
        ("doc hit rate", "doc_hit_rate", "", 4),
        ("page hit rate", "page_hit_rate", "", 4),
        ("answer match rate", "answer_match_rate", "", 4),
    ]
    for label, key, suffix, nd in rows:
        a = fmt(ds.get(key) if ds else None, suffix, nd)
        b = fmt(base.get(key) if base else None, suffix, nd)
        p(f"  {label:<26}{a:>19}{b:>19}")

    if ds and base and ds.get("total_wall_seconds") and base.get("total_wall_seconds"):
        speedup = base["total_wall_seconds"] / ds["total_wall_seconds"]
        p("-" * 68)
        p(f"  NRL speedup vs baseline   : {speedup:.2f}x wall-clock")
    p("=" * 68)

    text = "\n".join(out)
    print(text)
    return text


# ---------------------------------------------------------------------------
# Per-mode execution
# ---------------------------------------------------------------------------

def mode_args(args, mode: str):
    """Clone ``args`` with the per-mode toolset / indexing defaults applied.

    ``--toolsets`` stays an explicit override; when it is not given each mode
    picks its own default. Baseline never ingests — there is no NRL index.
    """
    m = copy.copy(args)
    m.mode = mode
    m.toolsets = args.toolsets or (
        "file,terminal" if mode == "baseline" else "document_search"
    )
    if mode == "baseline":
        m.skip_index = True
    return m


def run_mode(args, rows: list[dict]) -> tuple[list[QueryResult], float, float]:
    """Run every query for one mode. Returns (results, index_seconds, wall)."""
    wall_start = time.perf_counter()

    index_seconds = 0.0
    if not args.skip_index:
        log("phase 1/2: building / warming the NeMo Retriever index ...")
        warm = run_query(args, rows[0], idx=-1, reindex=args.reindex)
        index_seconds = warm.seconds
        if not warm.ok:
            log(f"  ! ingest/warm-up query failed: {warm.error}")
        else:
            log(f"  index ready in {index_seconds:.1f}s")

    log(f"phase 2/2: running {len(rows)} queries (concurrency={args.concurrency}) ...")
    results: list[QueryResult] = [None] * len(rows)  # type: ignore[list-item]
    done = 0

    def work(pair: tuple[int, dict]) -> None:
        nonlocal done
        i, row = pair
        results[i] = run_query(args, row, i)
        with _print_lock:
            done += 1
            mark = "." if results[i].ok else "x"
            print(f"\r  {done}/{len(rows)} {mark} ({results[i].seconds:.1f}s)",
                  end="", file=sys.stderr, flush=True)

    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(work, enumerate(rows)))
    else:
        for pair in enumerate(rows):
            work(pair)
    print("", file=sys.stderr)

    return results, index_seconds, time.perf_counter() - wall_start


def write_jsonl(path: Path, results: list[QueryResult], rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r, row in zip(results, rows):
            fh.write(json.dumps({
                "idx": r.idx, "query": r.query, "seconds": r.seconds, "ok": r.ok,
                "error": r.error, "doc_hit": r.doc_hit, "page_hit": r.page_hit,
                "answer": r.answer, "answer_match": r.answer_match,
                "expected": {"dir": row.get("dir"), "pdf": row.get("pdf"),
                             "page": row.get("page"), "modality": row.get("modality")},
                "hits": [{k: h.get(k) for k in ("source", "page", "score")} for h in r.hits],
            }, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    default_hermes = repo / ".venv" / "bin" / "hermes"

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--queries", default=DEFAULT_QUERIES)
    ap.add_argument("--limit", type=int, default=None, help="only run the first N queries")
    ap.add_argument("--concurrency", type=int, default=1, help="parallel CLI invocations")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--index-name", default="earnings_consulting",
                    help="persistent LanceDB table reused across queries")
    ap.add_argument("--reindex", action="store_true",
                    help="force a rebuild of the index during the ingest phase")
    ap.add_argument("--skip-index", action="store_true",
                    help="skip the ingest phase (index already built)")
    ap.add_argument("--mode", choices=("document_search", "baseline", "both"),
                    default="document_search",
                    help="document_search = with the NRL tool; baseline = same queries "
                         "with NO NRL tool; both = run each in turn and compare")
    ap.add_argument("--baseline", action="store_const", const="baseline", dest="mode",
                    help="shorthand for --mode baseline")
    ap.add_argument("--both", action="store_const", const="both", dest="mode",
                    help="shorthand for --mode both")
    ap.add_argument("--toolsets", default=None,
                    help="override the per-mode default ('document_search', "
                         "or 'file,terminal' in baseline mode)")
    ap.add_argument("--out-dir", default=None,
                    help="directory to save per-mode results: <mode>.jsonl, "
                         "<mode>.summary.txt, <mode>.summary.json (+ comparison.* for --mode both)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--provider", default=None)
    ap.add_argument("--reasoning", default=None)
    ap.add_argument("--hermes", default=str(default_hermes if default_hermes.exists() else "hermes"))
    ap.add_argument("--cwd", default=str(repo))
    ap.add_argument("--timeout", type=int, default=600, help="per-query timeout in seconds")
    ap.add_argument("--out", default=None, help="write per-query results to this JSONL file")
    ap.add_argument("--dry-run", action="store_true", help="print the first command and exit")
    args = ap.parse_args()

    args.corpus = str(Path(args.corpus).expanduser().resolve())

    rows = load_queries(Path(args.queries).expanduser(), args.limit)
    if not rows:
        print("error: no queries found in CSV", file=sys.stderr)
        return 2

    modes = ["document_search", "baseline"] if args.mode == "both" else [args.mode]

    if args.dry_run:
        for mode in modes:
            margs = mode_args(args, mode)
            prompt = build_prompt(margs, rows[0]["query"])
            print(f"# --- {MODE_LABEL[mode]} ---")
            print(" ".join(repr(c) if " " in c else c for c in build_cmd(margs, prompt)))
        return 0

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        log(f"saving results to {out_dir}")

    log(f"counting pages in {args.corpus} ...")
    total_pages, n_docs, page_method = count_pages(Path(args.corpus))
    if not total_pages:
        total_pages = pages_from_csv(rows)
        page_method = "csv ground-truth lower bound"
    log(f"  {n_docs} documents, {total_pages} pages ({page_method})")

    summaries: list[dict] = []
    exit_code = 0

    for mode in modes:
        margs = mode_args(args, mode)
        preflight(margs)
        log(f"\n=== mode: {mode} ({MODE_LABEL[mode]}) ===")

        results, index_seconds, wall = run_mode(margs, rows)
        text = report(margs, results, total_pages, page_method, n_docs, index_seconds, wall)
        summary = metrics(margs, results, total_pages, page_method, n_docs, index_seconds, wall)
        summaries.append(summary)
        if not all(r.ok for r in results):
            exit_code = 1

        jsonl_path = (out_dir / f"{mode}.jsonl") if out_dir else (
            Path(args.out) if (args.out and len(modes) == 1) else None
        )
        if jsonl_path:
            write_jsonl(jsonl_path, results, rows)
            log(f"  per-query results -> {jsonl_path}")
        if out_dir:
            (out_dir / f"{mode}.summary.txt").write_text(text + "\n", encoding="utf-8")
            (out_dir / f"{mode}.summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            log(f"  summary -> {out_dir / f'{mode}.summary.txt'}")

    if len(summaries) > 1:
        ctext = compare_report(summaries)
        if out_dir:
            (out_dir / "comparison.txt").write_text(ctext + "\n", encoding="utf-8")
            (out_dir / "comparison.json").write_text(
                json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            log(f"  comparison -> {out_dir / 'comparison.txt'}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
