"""Knowledge graph inspection and system status endpoints."""

from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from clinical_graph.api.deps import get_knowledge_manager
from clinical_graph.graph.manager import KnowledgeManager

graph_router = APIRouter(prefix="/v1/graph", tags=["graph"])
system_router = APIRouter(prefix="/v1/system", tags=["system"])


@system_router.get("/status")
async def get_system_status(request: Request) -> dict[str, bool]:
    """Return runtime configuration flags needed by the web UI."""
    settings = request.app.state.settings
    return {"has_openai_api_key": bool(settings.openai_api_key)}


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


__all__ = ["graph_router", "system_router"]
