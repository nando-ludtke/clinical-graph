"""Knowledge graph inspection and system status endpoints."""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from clinical_graph.api.deps import get_knowledge_manager
from clinical_graph.graph.manager import KnowledgeManager
from clinical_graph.guardrails.prompt_guard import ClinicalGuardClient

graph_router = APIRouter(prefix="/v1/graph", tags=["graph"])
system_router = APIRouter(prefix="/v1/system", tags=["system"])
benchmarks_router = APIRouter(prefix="/v1/benchmarks", tags=["system"])


@system_router.get("/status")
async def get_system_status(request: Request) -> dict[str, bool | int]:
    """Return runtime configuration flags needed by the web UI."""
    settings = request.app.state.settings
    guard_ok = (
        await ClinicalGuardClient(settings.clinical_guard_url).check_health()
        if settings.clinical_guard_url
        else False
    )
    return {
        "has_openai_api_key": bool(settings.openai_api_key),
        "prompt_guard_reachable": guard_ok,
        "prompt_guard_poll_seconds": settings.clinical_guard_health_poll_seconds,
        "auto_prune_non_frozen_hours": settings.auto_prune_non_frozen_hours,
    }


def _compact_run(run: dict[str, Any]) -> dict[str, Any]:
    """Strip bulky context and citation arrays from a single benchmark run."""
    return {
        "requested_mode": run.get("requested_mode"),
        "chosen_mode": run.get("chosen_mode"),
        "citation_count": run.get("citation_count"),
        "seed_node_count": len(run.get("seed_nodes") or []),
        "traversed_edge_count": len(run.get("traversed_edges") or []),
        "search_queries": run.get("search_queries") or [],
        "latency_ms": run.get("latency_ms"),
        "total_tokens": run.get("total_tokens"),
        "cost_usd": run.get("cost_usd"),
        "answer": run.get("answer"),
    }


def _collect_unit_evals(results_dir: Path) -> list[dict[str, Any]]:
    """Collect the latest summary for each unit eval suite from eval/results."""
    if not results_dir.is_dir():
        return []
    specs = [
        ("unit_triggers_*.json", "Web Triggers"),
        ("unit_retrieval_*.json", "Local Retrieval"),
        ("unit_web_search_*.json", "1-Shot Web Search"),
        ("unit_synthesis_*.json", "Answer Synthesis"),
    ]
    summaries: list[dict[str, Any]] = []
    for pattern, label in specs:
        files = sorted(results_dir.glob(pattern), key=lambda p: p.name, reverse=True)
        if not files:
            continue
        latest = files[0]
        try:
            payload = json.loads(latest.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = payload.get("results") or []
        latencies = [
            float(r["latency_ms"])
            for r in rows
            if isinstance(r, dict) and isinstance(r.get("latency_ms"), (int, float))
        ]
        avg_ms = round(sum(latencies) / len(latencies), 1) if latencies else None
        entry: dict[str, Any] = {
            "label": label,
            "filename": latest.name,
            "avg_latency_ms": avg_ms,
        }
        if "passed" in payload and "total" in payload:
            entry["passed"] = payload["passed"]
            entry["total"] = payload["total"]
        else:
            entry["cases"] = len(rows)
            selected = [
                int(r["selected_count"])
                for r in rows
                if isinstance(r, dict) and isinstance(r.get("selected_count"), int)
            ]
            entry["avg_sources"] = round(sum(selected) / len(selected), 1) if selected else None
        summaries.append(entry)
    return summaries


@benchmarks_router.get("")
async def get_benchmarks() -> dict[str, Any]:
    """Return the latest evaluation benchmark summary and compact per-query runs."""
    results_dir = Path("eval/results")
    if not results_dir.is_dir():
        results_dir = Path(__file__).resolve().parents[3] / "eval" / "results"

    unit_evals = _collect_unit_evals(results_dir)
    eval_files = (
        sorted(results_dir.glob("eval_*.json"), key=lambda p: p.name, reverse=True)
        if results_dir.is_dir()
        else []
    )
    if not eval_files:
        return {"available": False, "summary": None, "results": [], "unit_evals": unit_evals}

    latest_path = eval_files[0]
    data = json.loads(latest_path.read_text(encoding="utf-8"))

    compact_results: list[dict[str, Any]] = []
    for item in data.get("results", []):
        raw_runs = item.get("runs") or {}
        compact_runs = {
            mode: _compact_run(raw_runs[mode])
            for mode in ("rag", "hybrid", "auto")
            if isinstance(raw_runs.get(mode), dict)
        }
        compact_results.append(
            {
                "id": item.get("id"),
                "query": item.get("query"),
                "expected_mode": item.get("expected_mode"),
                "rationale": item.get("rationale"),
                "auto_matches_expected": item.get("auto_matches_expected"),
                "runs": compact_runs,
            }
        )

    return {
        "available": True,
        "filename": latest_path.name,
        "timestamp": data.get("timestamp"),
        "queries_file": data.get("queries_file"),
        "summary": data.get("summary"),
        "results": compact_results,
        "unit_evals": unit_evals,
    }


@graph_router.get("")
async def get_knowledge_graph(
    include_isolated: bool = Query(default=False),
    manager: KnowledgeManager = Depends(get_knowledge_manager),
) -> dict[str, list[dict[str, Any]]]:
    """Return nodes and directed edges from the in-memory knowledge graph."""
    nx_graph = manager.graph.graph

    document_node_ids: set[str] = set()
    for _, data in nx_graph.nodes(data=True):
        node = data.get("node")
        if node is not None and node.node_type == "document":
            document_node_ids.add(node.node_id)

    edges: list[dict[str, Any]] = []
    connected_node_ids: set[str] = set()
    for u, v, data in nx_graph.edges(data=True):
        edge = data.get("edge")
        if edge is None:
            continue
        if edge.source in document_node_ids or edge.target in document_node_ids:
            continue
        connected_node_ids.add(edge.source)
        connected_node_ids.add(edge.target)
        edges.append(
            {
                "id": f"{u}->{v}",
                "source": edge.source,
                "target": edge.target,
                "relation": edge.relation,
                "relationship_description": edge.relationship_description,
                "weight": edge.weight,
                "doc_ids": sorted(edge.doc_ids),
                "evidence": edge.evidence,
            }
        )

    nodes: list[dict[str, Any]] = []
    for _, data in nx_graph.nodes(data=True):
        node = data.get("node")
        if node is None or node.node_type == "document":
            continue
        if not include_isolated and node.node_id not in connected_node_ids:
            continue
        nodes.append(
            {
                "node_id": node.node_id,
                "canonical_name": node.canonical_name,
                "node_type": node.node_type,
                "aliases": list(node.aliases),
                "description": node.description,
                "doc_ids": sorted(node.doc_ids),
            }
        )

    return {"nodes": nodes, "edges": edges}


__all__ = ["benchmarks_router", "graph_router", "system_router"]
