"""Dump each source's complete field inventory to knowledge/<pack>/schemas/<source>.yaml.

The retriever's flattener stops at ARRAY<STRUCT<...>> because a dotted path into an array
is not valid SQL; this script descends arrays and records explode_path + array_depth per
leaf so the inventory is complete. Runs _reconcile_nested_types before flattening because
full_data_type in information_schema can serve a stale nested schema.
With --measure, counts non-null occurrences per leaf over a bounded partition window.

Output is one YAML per source under knowledge/<pack>/schemas/. Re-running merges: existing
descriptions are preserved, vanished leaves moved to removed:. Read-only: SELECT / DESCRIBE
/ information_schema / field_caps only.

Usage:
    set -a; source .afir_env; set +a
    python -u scripts/generate_source_schemas.py                      # configured pack
    python -u scripts/generate_source_schemas.py --pack <pack_dir>    # any installed pack
    python -u scripts/generate_source_schemas.py --source <name> --measure
    python -u scripts/generate_source_schemas.py --measure --days 2   # cheaper window

The read-only single-question counterpart is src/knowledge/pack_probe.py.
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import yaml  # noqa: E402  (after sys.path setup)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("schemagen")


def schemas_dir(pack_name: str) -> Path:
    """Where this pack's inventory lives. Resolved from the pack, never a constant."""
    return REPO_ROOT / "knowledge" / pack_name / "schemas"


# Higher than the retriever's caps (_MAX_STRUCT_DEPTH=3, _MAX_LEAVES_PER_TABLE=60):
# those protect a prompt budget; this artifact is the RAG corpus, where completeness matters.
MAX_DEPTH = 12
MAX_LEAVES = 5000

# An unpartitioned table is measured only when sizeInBytes from DESCRIBE DETAIL is
# positive and below this threshold; an unbounded aggregate on anything larger is a full scan.
SMALL_TABLE_BYTES = 2 * 1024**3

# Budget is divided between the pattern's concrete indices, so this is the floor on
# per-index docs as much as a cap on breadth.
_MAX_SAMPLED_INDICES = 40


# ---------------------------------------------------------------------------
# Type-string parsing. Same grammar as the retriever's _split_top_level, but this
# walker DESCENDS arrays and records the explode path instead of stopping.
# ---------------------------------------------------------------------------
def split_top_level(fields: str) -> List[str]:
    """Split a STRUCT/ARRAY field list on commas outside angle brackets."""
    parts, depth, cur = [], 0, []
    for ch in fields:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def flatten_all(
    path: str,
    type_str: str,
    depth: int = 0,
    explode_path: Optional[str] = None,
    array_depth: int = 0,
) -> List[Dict[str, Any]]:
    """Flatten a column type into EVERY leaf, descending arrays and maps.

    Returns dicts with:
      path         dotted path from the column root, array hops included by name.
      type         the leaf's scalar type (or the compound type at a depth cap).
      explode_path the ARRAY path that must be `explode()`d to reach this leaf, or None
                   when the leaf is plain dot-accessible. This is the fact the retriever's
                   flattener cannot express and the reason nested leaves went unseen.
      array_depth  how many array hops deep (>1 means nested explodes).
      kind         scalar | array | map | struct — what the leaf itself is.
    """
    t = (type_str or "").strip()
    up = t.upper()

    if depth >= MAX_DEPTH:
        return [_leaf(path, t, explode_path, array_depth, "struct")]

    if up.startswith("ARRAY<") and t.endswith(">"):
        inner = t[len("ARRAY<") : -1].strip()
        # The array itself is a leaf too: a query may return it whole, and knowing it is an
        # array is what tells the generator it needs an explode.
        out = [_leaf(path, t, explode_path, array_depth, "array")]
        out.extend(
            flatten_all(path, inner, depth + 1, explode_path=path, array_depth=array_depth + 1)
        )
        return out

    if up.startswith("MAP<") and t.endswith(">"):
        # A map's keys are data, not schema; there is nothing to enumerate.
        return [_leaf(path, t, explode_path, array_depth, "map")]

    if up.startswith("STRUCT<") and t.endswith(">"):
        body = t[len("STRUCT<") : -1]
        leaves: List[Dict[str, Any]] = []
        for field in split_top_level(body):
            if ":" not in field:
                continue
            fname, ftype = field.split(":", 1)
            leaves.extend(
                flatten_all(
                    f"{path}.{fname.strip()}",
                    ftype.strip(),
                    depth + 1,
                    explode_path,
                    array_depth,
                )
            )
        return leaves or [_leaf(path, t, explode_path, array_depth, "struct")]

    return [_leaf(path, t, explode_path, array_depth, "scalar")]


def _leaf(path, t, explode_path, array_depth, kind) -> Dict[str, Any]:
    leaf: Dict[str, Any] = {"path": path, "type": t, "kind": kind}
    if explode_path:
        leaf["explode_path"] = explode_path
        leaf["array_depth"] = array_depth
    return leaf


# ---------------------------------------------------------------------------
# Databricks
# ---------------------------------------------------------------------------
async def dump_databricks(retr, source, measure: bool, days: int) -> Dict[str, Any]:
    cat, sch = retr.catalog, retr.schema
    tables = list(retr.tables or (source.endpoints or {}).get("tables") or [])
    if not tables:
        log.warning("%s: no tables declared; skipping", source.name)
        return {}

    quoted = ", ".join(f"'{t}'" for t in tables)
    rows = await retr._execute_sql(
        "SELECT table_name, column_name, data_type, full_data_type, partition_index, "
        "comment, ordinal_position "
        f"FROM {cat}.information_schema.columns "
        f"WHERE table_schema = '{sch}' AND table_name IN ({quoted}) "
        "ORDER BY table_name, ordinal_position"
    )
    # full_data_type can serve stale nested types; reuse the retriever's reconciliation
    # so the inventory and the query generator always read the same schema.
    await retr._reconcile_nested_types(rows, sch)

    out_tables: Dict[str, Any] = {}
    for row in rows:
        tname = row["table_name"]
        tbl = out_tables.setdefault(
            tname,
            {
                "fully_qualified_name": f"{cat}.{sch}.{tname}",
                "partition_columns": [],
                "columns": {},
            },
        )
        col = row["column_name"]
        type_str = row.get("full_data_type") or row.get("data_type") or ""
        if row.get("partition_index") is not None:
            tbl["partition_columns"].append(col)
        leaves = flatten_all(col, type_str)
        if len(leaves) > MAX_LEAVES:
            log.warning("%s.%s: %d leaves, truncating", tname, col, len(leaves))
            leaves = leaves[:MAX_LEAVES]
        entry: Dict[str, Any] = {"type": type_str, "leaves": leaves}
        if row.get("comment"):
            entry["backend_comment"] = row["comment"]
        tbl["columns"][col] = entry

    # Clustering is not in information_schema; fetched separately for the annotation.
    for tname, tbl in out_tables.items():
        try:
            detail = await retr._execute_sql(f"DESCRIBE DETAIL {cat}.{sch}.{tname}")
            if detail:
                d = detail[0]
                tbl["physical"] = {
                    "num_files": d.get("numFiles"),
                    "size_bytes": d.get("sizeInBytes"),
                    "clustering_columns": d.get("clusteringColumns"),
                    "partition_columns_reported": d.get("partitionColumns"),
                }
        except Exception as e:  # noqa: BLE001  a VIEW has no DESCRIBE DETAIL
            log.info("%s: DESCRIBE DETAIL unavailable (%s)", tname, type(e).__name__)

    if measure:
        for tname, tbl in out_tables.items():
            await measure_population(retr, cat, sch, tname, tbl, days)
    return out_tables


async def measure_population(retr, cat, sch, tname, tbl, days: int) -> None:
    """Count non-null occurrences per leaf over a bounded partition window.

    A rule keyed on an always-empty leaf fails the same way as one keyed on an absent leaf;
    only a count separates them.

    Leaves are batched into wide aggregates (not one query each). Array leaves are counted
    through the explode they require.

    Unpartitioned tables are not measured unless sizeInBytes shows they are small; an
    unbounded aggregate is a full scan. The skip is recorded in measured.skipped_reason.
    """
    parts = tbl.get("partition_columns") or []
    if not parts:
        size = float((tbl.get("physical") or {}).get("size_bytes") or 0)
        if size <= 0 or size > SMALL_TABLE_BYTES:
            tbl.setdefault("measured", {})["skipped_reason"] = (
                "no partition column to bound and the table is not demonstrably small "
                f"(size_bytes={(tbl.get('physical') or {}).get('size_bytes')}); an "
                "unbounded aggregate here is a full scan, so population is left unmeasured "
                "rather than paid for. Measure specific leaves with a targeted probe."
            )
            log.warning("%s: unpartitioned and not small — population not measured", tname)
            return
        where = "1=1"  # demonstrably small: a full scan is cheap and bounded
    else:
        end = date.today()
        start = end - timedelta(days=days)
        # Bound every partition column; type is unknown here so DATE literals are tried and
        # a failure downgrades to an unmeasured table rather than a wrong number.
        where = " AND ".join(
            f"{p} >= DATE'{start.isoformat()}' AND {p} <= DATE'{end.isoformat()}'"
            for p in parts
        )

    plain: List[Tuple[str, Dict]] = []
    by_explode: Dict[str, List[Tuple[str, Dict]]] = {}
    for col in tbl["columns"].values():
        for leaf in col["leaves"]:
            if leaf["kind"] in ("struct",):
                continue
            if leaf.get("explode_path"):
                if leaf.get("array_depth", 1) > 1:
                    continue  # nested explodes: not worth a generic probe
                by_explode.setdefault(leaf["explode_path"], []).append((leaf["path"], leaf))
            else:
                plain.append((leaf["path"], leaf))

    async def run_batch(sel_leaves, lateral: str = "", alias_root: str = ""):
        if not sel_leaves:
            return
        for i in range(0, len(sel_leaves), 60):
            chunk = sel_leaves[i : i + 60]
            sel = ", ".join(
                f"count({_expr(p, alias_root)}) AS c{j}" for j, (p, _) in enumerate(chunk)
            )
            sql = (
                f"SELECT count(*) AS total, {sel} "
                f"FROM {cat}.{sch}.{tname} {lateral} WHERE {where}"
            )
            try:
                res = await asyncio.wait_for(retr._execute_sql(sql), timeout=900)
            except Exception as e:  # noqa: BLE001
                log.warning("%s: population batch failed (%s)", tname, type(e).__name__)
                return
            if not res:
                return
            total = float(res[0].get("total") or 0)
            tbl.setdefault("measured", {})["rows_in_window"] = int(total)
            tbl["measured"]["window_days"] = days
            for j, (_, leaf) in enumerate(chunk):
                n = float(res[0].get(f"c{j}") or 0)
                leaf["populated_pct"] = round(100.0 * n / total, 2) if total else None

    def _expr(p: str, alias_root: str) -> str:
        if alias_root:
            # Rewrite `outer.collection.leaf` -> `x.leaf` (the exploded alias)
            return "x." + p[len(alias_root) + 1 :] if p.startswith(alias_root + ".") else p
        return p

    await run_batch(plain)
    for arr_path, leaves in by_explode.items():
        await run_batch(
            leaves,
            lateral=f"LATERAL VIEW OUTER explode({arr_path}) _x AS x",
            alias_root=arr_path,
        )


# ---------------------------------------------------------------------------
# Elasticsearch
# ---------------------------------------------------------------------------
async def dump_elasticsearch(retr, source, sample_size: int) -> Dict[str, Any]:
    """Field inventory for an ES-backed source.

    Two routes:
      * direct client (ElasticsearchRetriever): field_caps returns the authoritative mapping
        with searchable/aggregatable flags per field.
      * Kibana gateway (KibanaRetriever): no mapping endpoint behind /internal/search/es,
        so fields are sampled from documents. Uncapped depth (the retriever's discovery caps
        at 3). Sampling is stratified per concrete index so a pattern spanning multiple
        document shapes covers all of them; a flat match_all can return docs from only one.

    present_in_sampled_pct weights each index equally, not each document.
    """
    # Route 1: a direct ES client exposes the mapping.
    if hasattr(retr, "_get_client"):
        try:
            resp = await retr._get_client().field_caps(index=retr.index, fields="*")
            body = resp.body if hasattr(resp, "body") else resp
            fields: Dict[str, Any] = {}
            for name, types in (body.get("fields") or {}).items():
                entry: Dict[str, Any] = {"types": {}}
                for tname, meta in (types or {}).items():
                    entry["types"][tname] = {
                        "searchable": meta.get("searchable"),
                        "aggregatable": meta.get("aggregatable"),
                    }
                fields[name] = entry
            return {
                retr.index: {
                    "index_pattern": retr.index,
                    "discovery": "field_caps (authoritative mapping)",
                    "fields": fields,
                }
            }
        except Exception as e:  # noqa: BLE001  fall through to sampling
            log.info("%s: field_caps unavailable (%s); sampling", source.name, e)

    # Route 2: the Kibana gateway — sample documents, uncapped in depth.
    if not hasattr(retr, "_search"):
        log.warning("%s: no field_caps and no _search; cannot dump", source.name)
        return {}
    hits, n_indices = await _stratified_sample(retr, source, sample_size)
    if not hits:
        log.warning("%s: sample returned 0 docs", source.name)
        return {}

    counts: Dict[str, int] = {}
    types: Dict[str, set] = {}

    def walk(obj: Any, prefix: str) -> None:
        """Union dotted paths across the sample, recording each leaf's JSON type(s).

        No depth cap (the retriever's is 3); lists are descended because an ES field
        holding an array of objects is queried by the same dotted path as a single object.
        """
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{prefix}.{k}" if prefix else k
                counts[p] = counts.get(p, 0) + 1
                types.setdefault(p, set()).add(_json_kind(v))
                walk(v, p)
        elif isinstance(obj, list):
            for item in obj:
                walk(item, prefix)

    for hit in hits:
        walk(hit.get("_source") or {}, "")

    n = len(hits)
    fields = {
        name: {
            "json_types": sorted(types.get(name, set())),
            "present_in_sampled_pct": round(100.0 * counts[name] / n, 2),
        }
        for name in sorted(counts)
    }
    return {
        retr.index: {
            "index_pattern": retr.index,
            "discovery": (
                "sampled from documents via the Kibana gateway (no mapping endpoint "
                "behind /internal/search/es), STRATIFIED per concrete index so a pattern "
                "spanning several document shapes shows all of them. A field missing here "
                "was absent from the sampled documents — that is NOT the same as absent "
                "from the mapping; and its percentage is a fraction of the stratified "
                "sample, which weights each index equally rather than each document."
            ),
            "sampled_docs": n,
            "sampled_indices": n_indices,
            "fields": fields,
        }
    }


async def _stratified_sample(retr, source, sample_size: int) -> Tuple[List[Dict], int]:
    """`sample_size` documents spread over the pattern's concrete indices.

    Index list via a `terms` agg on `_index`, ordered newest first (descending `_key` sorts
    a date-suffixed name reverse-chronologically). Budget divided between buckets.
    Falls back to a flat match_all when the aggregation returns nothing.
    """
    per_index = 0
    buckets: List[Dict] = []
    try:
        agg = await retr._search(
            {
                "size": 0,
                "query": {"match_all": {}},
                "aggs": {
                    "idx": {
                        "terms": {
                            "field": "_index",
                            "size": _MAX_SAMPLED_INDICES,
                            "order": {"_key": "desc"},
                        }
                    }
                },
            }
        )
        idx = (agg.get("aggregations") or {}).get("idx") or {}
        buckets = idx.get("buckets") or []
    except Exception as e:  # noqa: BLE001
        log.info("%s: per-index bucketing unavailable (%s)", source.name, e)
    if buckets:
        per_index = max(1, sample_size // len(buckets))
    hits: List[Dict] = []
    seen_indices = 0
    for bucket in buckets:
        name = bucket.get("key")
        if not name:
            continue
        try:
            raw = await retr._search(
                {
                    "size": per_index,
                    "query": {"term": {"_index": name}},
                }
            )
        except Exception as e:  # noqa: BLE001
            log.warning("%s: sampling %s failed (%s)", source.name, name, e)
            continue
        got = (raw.get("hits") or {}).get("hits") or []
        if got:
            seen_indices += 1
            hits.extend(got)
    if hits:
        log.info(
            "%s: sampled %d docs across %d indices",
            source.name,
            len(hits),
            seen_indices,
        )
        return hits, seen_indices
    log.warning(
        "%s: per-index sampling yielded nothing; falling back to a flat sample "
        "(one document shape may be missed)",
        source.name,
    )
    try:
        raw = await retr._search({"size": sample_size, "query": {"match_all": {}}})
    except Exception as e:  # noqa: BLE001
        log.warning("%s: document sampling failed (%s)", source.name, e)
        return [], 0
    return ((raw.get("hits") or {}).get("hits") or []), 1


def _json_kind(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__


# ---------------------------------------------------------------------------
# Merge with the existing hand annotation
# ---------------------------------------------------------------------------
def merge_annotations(new_doc: Dict[str, Any], path: Path) -> Dict[str, Any]:
    """Carry hand-written description: values forward; move vanished leaves to removed:.

    Runs even with no prior file; empty description: '' slots are the deliverable so the
    difference between "documented as irrelevant" and "not yet annotated" is visible.
    """
    old: Dict[str, Any] = {}
    if path.exists():
        try:
            old = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as e:  # noqa: BLE001
            log.warning("%s unreadable, not merging (%s)", path, e)
            old = {}

    descs: Dict[str, str] = {}

    # Harvest only from the two shapes that carry an annotation; a generic walk would
    # silently attach descriptions to the wrong field.
    for tbl in (old.get("tables") or {}).values():
        if not isinstance(tbl, dict):
            continue
        for col in (tbl.get("columns") or {}).values():  # Databricks: leaves carry `path`
            if not isinstance(col, dict):
                continue
            if col.get("description"):
                descs[f"__column__{id(col)}"] = col["description"]  # placeholder, see below
            for leaf in col.get("leaves") or []:
                if isinstance(leaf, dict) and leaf.get("path") and leaf.get("description"):
                    descs[str(leaf["path"])] = leaf["description"]
        for fname, f in (tbl.get("fields") or {}).items():  # Elasticsearch: keyed by name
            if isinstance(f, dict) and f.get("description"):
                descs[str(fname)] = f["description"]
    # Column-level prose is keyed by its own name, not by object identity.
    for tname, tbl in (old.get("tables") or {}).items():
        if not isinstance(tbl, dict):
            continue
        for cname, col in (tbl.get("columns") or {}).items():
            if isinstance(col, dict) and col.get("description"):
                descs[f"{tname}::{cname}"] = col["description"]
    for k in [k for k in descs if k.startswith("__column__")]:
        del descs[k]
    # Table-level prose ("what this table IS"), keyed by table name.
    table_notes = {
        tname: tbl["description"]
        for tname, tbl in (old.get("tables") or {}).items()
        if isinstance(tbl, dict) and tbl.get("description")
    }

    seen = set()
    for tname, tbl in (new_doc.get("tables") or {}).items():
        tbl["description"] = table_notes.get(tname, tbl.get("description", ""))
        for cname, col in (tbl.get("columns") or {}).items():
            col["description"] = descs.get(f"{tname}::{cname}", col.get("description", ""))
            for leaf in col.get("leaves") or []:
                p = str(leaf.get("path") or "")
                if not p:
                    continue
                seen.add(p)
                leaf["description"] = descs.get(p, leaf.get("description", ""))
        for fname, f in (tbl.get("fields") or {}).items():
            seen.add(str(fname))
            if isinstance(f, dict):
                f["description"] = descs.get(str(fname), f.get("description", ""))

    # Only leaf paths participate in the vanished-path check; tbl::col and table-name
    # keys are a different namespace and would all appear missing.
    gone = {p: d for p, d in descs.items() if "::" not in p and p not in seen}
    if gone:
        new_doc["removed"] = {
            "note": (
                "Paths that carried a hand-written description but are no longer in the "
                "live schema. Kept because a field vanishing from a production table is "
                "itself worth reviewing — delete only once that is understood."
            ),
            "paths": gone,
        }
        log.warning("%s: %d annotated path(s) no longer in the live schema", path.name, len(gone))
    return new_doc


def leaf_count(doc: Dict[str, Any]) -> int:
    n = 0
    for tbl in (doc.get("tables") or {}).values():
        for col in (tbl.get("columns") or {}).values():
            n += len(col.get("leaves") or [])
        n += len(tbl.get("fields") or {})
    return n


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pack",
        default=None,
        help="pack directory name under knowledge/ (default: knowledge.pack_dir from "
        "main_config.yaml — the pack this deployment actually investigates with)",
    )
    ap.add_argument("--source", action="append", help="limit to these source names")
    ap.add_argument("--measure", action="store_true", help="count non-null per leaf")
    ap.add_argument("--days", type=int, default=3, help="measurement window (days)")
    ap.add_argument(
        "--es-sample",
        type=int,
        default=500,
        help="documents to sample behind the Kibana gateway, DIVIDED between the "
        "pattern's concrete indices (no mapping endpoint there, so the inventory is a "
        "sample and a bigger one is a better one)",
    )
    args = ap.parse_args()

    from main import load_config
    from log_retrieval import LogRetrievalEngine
    from utils.databricks_auth import try_build_auth
    from utils.llm_client import LLMClient
    from utils.paths import config_path, knowledge_pack_dir
    from knowledge.pack import load_knowledge_pack

    mc = load_config(config_path("main_config.yaml"))
    lc = load_config(config_path("llm_config.yaml"))
    auth = try_build_auth()
    llm = LLMClient(lc, auth=auth if auth else None)
    # Default to the configured pack so schemas match the sources this deployment can reach.
    pack_name = args.pack or str(
        (mc.get("knowledge", {}) or {}).get("pack_dir", "") or ""
    ).strip()
    if not pack_name:
        ap.error(
            "no pack to generate for: pass --pack <dir>, or set knowledge.pack_dir in "
            "main_config.yaml"
        )
    pack = load_knowledge_pack(knowledge_pack_dir(pack_name))
    eng = LogRetrievalEngine(mc["log_sources"], llm, auth=auth, knowledge_pack=pack)

    out_dir = schemas_dir(pack_name)
    print(f"pack: {pack_name} -> {out_dir.relative_to(REPO_ROOT)}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(args.source or [])
    written = []

    for source in pack.sources:
        if wanted and source.name not in wanted:
            continue
        retr = eng.retrievers.get(source.name)
        if retr is None:
            log.info("%s: no retriever built (creds absent / unsupported kind)", source.name)
            continue
        kind = source.kind()
        print(f"\n=== {source.name} ({kind}) ===", flush=True)
        try:
            if kind == "databricks_uc":
                retr.max_poll_attempts = max(getattr(retr, "max_poll_attempts", 0), 400)
                tables = await dump_databricks(retr, source, args.measure, args.days)
            elif kind == "elasticsearch":
                tables = await dump_elasticsearch(retr, source, args.es_sample)
            else:
                log.info("%s: kind '%s' has no dumper yet", source.name, kind)
                continue
        except Exception as e:  # noqa: BLE001  one source must not stop the sweep
            log.warning("%s: dump failed (%s: %s)", source.name, type(e).__name__, e)
            continue
        if not tables:
            continue

        doc: Dict[str, Any] = {
            "source": source.name,
            "kind": kind,
            "generated_by": "scripts/generate_source_schemas.py",
            "note": (
                "GENERATED field inventory + HAND-WRITTEN descriptions. The structure "
                "(paths, types, explode_path, populated_pct) is discovered from the live "
                "backend and is overwritten on every regeneration; `description:` is "
                "written by hand and preserved. Arrays ARE descended here, unlike the "
                "retriever's query-generation schema, which stops at an ARRAY<STRUCT<...>> "
                "because a dotted path into an array is not valid SQL. A leaf with an "
                "`explode_path` is reachable only through that explode."
            ),
            "tables": tables,
        }
        out = out_dir / f"{source.name}.yaml"
        doc = merge_annotations(doc, out)
        out.write_text(
            yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100),
            encoding="utf-8",
        )
        n = leaf_count(doc)
        written.append((source.name, n))
        print(f"  -> {out.relative_to(REPO_ROOT)}  ({n} leaves)", flush=True)

    print(f"\n=== wrote {len(written)} file(s) ===", flush=True)
    for name, n in written:
        print(f"  {name:34s} {n:6d} leaves", flush=True)
    await eng.close()


if __name__ == "__main__":
    asyncio.run(main())
