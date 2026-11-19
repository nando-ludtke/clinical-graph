"""Hybrid Knowledge Graph wrapping NetworkX DiGraph and LocalVectorStore."""

from collections import deque
import json
from pathlib import Path
import re
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field

from clinical_graph.graph.vector_store import LocalVectorStore

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _normalize_term(term: str) -> str:
    return " ".join(term.strip().lower().split())


class GraphNode(BaseModel):
    """Clinical entity or document node stored in the knowledge graph."""

    node_id: str
    canonical_name: str
    node_type: str = "entity"
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    doc_ids: set[str] = Field(default_factory=set)


class GraphEdge(BaseModel):
    """Directed clinical relationship edge with document provenance and evidence."""

    source: str
    target: str
    relation: str
    weight: float = 5.0
    relationship_description: str = ""
    doc_ids: set[str] = Field(default_factory=set)
    evidence: list[dict[str, str]] = Field(default_factory=list)


class ClinicalKnowledgeGraph:
    """Hybrid knowledge graph combining NetworkX DiGraph, alias index, and vector store."""

    def __init__(self, vector_store: LocalVectorStore | None = None) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._vector_store: LocalVectorStore = vector_store or LocalVectorStore()
        self._alias_index: dict[str, set[str]] = {}

    @property
    def graph(self) -> nx.DiGraph:
        """Return the underlying NetworkX directed graph."""
        return self._graph

    @property
    def vector_store(self) -> LocalVectorStore:
        """Return the underlying local vector store."""
        return self._vector_store

    def get_node(self, node_id: str) -> GraphNode | None:
        """Return the ``GraphNode`` for ``node_id``, or ``None`` if absent."""
        data = self._graph.nodes.get(node_id)
        if data is None:
            return None
        return data["node"]

    def resolve_existing_node_id(
        self,
        canonical_name: str,
        aliases: list[str] | None = None,
    ) -> str | None:
        """Resolve an entity's canonical name or aliases to an existing non-document node ID."""
        terms = [canonical_name, *(aliases or [])]
        for term in terms:
            norm = _normalize_term(term)
            if not norm:
                continue
            candidate_ids = self._alias_index.get(norm)
            if not candidate_ids:
                continue
            matches = sorted(
                nid
                for nid in candidate_ids
                if (node := self.get_node(nid)) is not None and node.node_type != "document"
            )
            if matches:
                return matches[0]
        return None

    def _upsert_node_in_graph(
        self,
        node_id: str,
        canonical_name: str,
        node_type: str = "entity",
        aliases: list[str] | None = None,
        description: str = "",
        doc_id: str | None = None,
    ) -> tuple[GraphNode, str]:
        incoming_aliases = [a.strip() for a in (aliases or []) if a.strip()]
        existing_data = self._graph.nodes.get(node_id)

        if existing_data is not None:
            node: GraphNode = existing_data["node"]
            self._unindex_node_aliases(node)
            merged_aliases = _merge_aliases(node.aliases, incoming_aliases)
            node.canonical_name = canonical_name or node.canonical_name
            node.node_type = node_type or node.node_type
            node.aliases = merged_aliases
            if description:
                node.description = description
            if doc_id is not None:
                node.doc_ids.add(doc_id)
        else:
            node = GraphNode(
                node_id=node_id,
                canonical_name=canonical_name,
                node_type=node_type,
                aliases=_merge_aliases([], incoming_aliases),
                description=description,
                doc_ids={doc_id} if doc_id is not None else set(),
            )
            self._graph.add_node(node_id, node=node)

        self._index_node_aliases(node)
        embed_text = (
            f"{node.canonical_name} | Aliases: {', '.join(node.aliases)} | {node.description}"
        )
        return node, embed_text

    @staticmethod
    def _build_node_label_text(canonical_name: str, aliases: list[str]) -> str:
        clean_aliases = [a for a in aliases if a and a.strip()]
        if clean_aliases:
            return f"{canonical_name} | Aliases: {', '.join(clean_aliases)}"
        return canonical_name

    def upsert_node(
        self,
        node_id: str,
        canonical_name: str,
        node_type: str = "entity",
        aliases: list[str] | None = None,
        description: str = "",
        doc_id: str | None = None,
        api_key: str | None = None,
    ) -> GraphNode:
        """Insert or merge a node, update alias index, and refresh its vector embeddings."""
        node, embed_text = self._upsert_node_in_graph(
            node_id=node_id,
            canonical_name=canonical_name,
            node_type=node_type,
            aliases=aliases,
            description=description,
            doc_id=doc_id,
        )
        self._vector_store.upsert(node_id, embed_text, api_key=api_key)
        if node.node_type != "document":
            label_text = self._build_node_label_text(node.canonical_name, node.aliases)
            self._vector_store.upsert_label(node.node_id, label_text, api_key=api_key)
        return node

    def upsert_nodes_batch(
        self,
        nodes_spec: list[dict[str, Any]],
        doc_id: str | None = None,
        api_key: str | None = None,
    ) -> list[GraphNode]:
        """Insert or merge multiple nodes and refresh their full and label-only embeddings."""
        if not nodes_spec:
            return []
        nodes: list[GraphNode] = []
        embed_by_id: dict[str, str] = {}
        label_by_id: dict[str, str] = {}
        for spec in nodes_spec:
            effective_doc_id = spec.get("doc_id") if spec.get("doc_id") is not None else doc_id
            node, embed_text = self._upsert_node_in_graph(
                node_id=spec["node_id"],
                canonical_name=spec["canonical_name"],
                node_type=spec.get("node_type", "entity"),
                aliases=spec.get("aliases"),
                description=spec.get("description", ""),
                doc_id=effective_doc_id,
            )
            nodes.append(node)
            embed_by_id[node.node_id] = embed_text
            if node.node_type != "document":
                label_by_id[node.node_id] = self._build_node_label_text(
                    node.canonical_name,
                    node.aliases,
                )

        self._vector_store.upsert_many(list(embed_by_id.items()), api_key=api_key)
        if label_by_id:
            self._vector_store.upsert_labels_batch(list(label_by_id.items()), api_key=api_key)
        return nodes

    def check_label_coverage(
        self,
        query_vectors: list[Any],
        *,
        threshold: float = 0.48,
        api_key: str | None = None,
    ) -> tuple[bool, float]:
        """Return ``(all_covered, min_best_score)`` against label-only embeddings."""
        if not self._vector_store._label_vectors or not query_vectors:
            return True, 1.0

        best_scores: list[float] = []
        for qvec in query_vectors:
            hits = self._vector_store.search_labels(
                "",
                top_k=1,
                query_vector=qvec,
                api_key=api_key,
            )
            best_scores.append(hits[0][1] if hits else 0.0)
        min_best = min(best_scores) if best_scores else 0.0
        return min_best >= threshold, min_best

    def upsert_edge(
        self,
        source: str,
        target: str,
        relation: str,
        doc_id: str,
        excerpt: str = "",
        chunk_id: str | None = None,
        weight: float = 5.0,
        relationship_description: str = "",
    ) -> GraphEdge | None:
        """Insert or update a directed edge with document provenance and evidence."""
        if source.strip().lower() == target.strip().lower():
            return None

        if source not in self._graph:
            self.upsert_node(source, canonical_name=source, doc_id=doc_id)
        if target not in self._graph:
            self.upsert_node(target, canonical_name=target, doc_id=doc_id)

        evidence_entry: dict[str, str] = {
            "doc_id": doc_id,
            "relation": relation,
            "excerpt": excerpt,
        }
        if chunk_id is not None:
            evidence_entry["chunk_id"] = chunk_id
        if relationship_description:
            evidence_entry["relationship_description"] = relationship_description

        if self._graph.has_edge(source, target):
            edge: GraphEdge = self._graph.edges[source, target]["edge"]
            edge.doc_ids.add(doc_id)
            edge.weight = max(edge.weight, weight)
            if relationship_description and len(relationship_description) > len(
                edge.relationship_description
            ):
                edge.relationship_description = relationship_description
            if evidence_entry not in edge.evidence:
                edge.evidence.append(evidence_entry)
        else:
            edge = GraphEdge(
                source=source,
                target=target,
                relation=relation,
                weight=weight,
                relationship_description=relationship_description,
                doc_ids={doc_id},
                evidence=[evidence_entry],
            )
            self._graph.add_edge(source, target, edge=edge)
        return edge

    def remove_document_provenance(self, doc_id: str) -> tuple[int, int, int]:
        """Remove all provenance for ``doc_id`` and prune orphaned edges, nodes, and vectors.

        Returns ``(removed_nodes, removed_edges, removed_embeddings)``.
        """
        removed_edges = 0
        edges_to_remove: list[tuple[str, str]] = []

        for u, v, data in self._graph.edges(data=True):
            edge: GraphEdge = data["edge"]
            if doc_id in edge.doc_ids:
                edge.doc_ids.discard(doc_id)
                edge.evidence = [e for e in edge.evidence if e.get("doc_id") != doc_id]
                if not edge.doc_ids:
                    edges_to_remove.append((u, v))

        for u, v in edges_to_remove:
            self._graph.remove_edge(u, v)
            removed_edges += 1

        nodes_to_remove: list[str] = []
        for nid, data in self._graph.nodes(data=True):
            node: GraphNode = data["node"]
            if doc_id in node.doc_ids:
                node.doc_ids.discard(doc_id)
                if not node.doc_ids:
                    nodes_to_remove.append(nid)

        removed_nodes = 0
        removed_embeddings = 0
        for nid in nodes_to_remove:
            if nid not in self._graph:
                continue
            incident_edges = self._graph.in_degree(nid) + self._graph.out_degree(nid)
            removed_edges += incident_edges

            node = self._graph.nodes[nid]["node"]
            self._unindex_node_aliases(node)
            if self._vector_store.delete(nid):
                removed_embeddings += 1
            self._graph.remove_node(nid)
            removed_nodes += 1

        removed_embeddings += self._vector_store.delete_by_doc_id(doc_id)
        return removed_nodes, removed_edges, removed_embeddings

    def find_seed_nodes(
        self,
        query: str,
        top_k: int = 5,
        allowed_doc_ids: set[str] | None = None,
        api_key: str | None = None,
        query_vector: Any = None,
    ) -> list[tuple[str, float]]:
        """Find top ``top_k`` seed nodes combining exact alias matching and vector similarity."""
        if top_k <= 0 or self._graph.number_of_nodes() == 0:
            return []

        candidate_node_ids: set[str] | None = None
        if allowed_doc_ids is not None:
            candidate_node_ids = {
                nid
                for nid, data in self._graph.nodes(data=True)
                if data["node"].doc_ids & allowed_doc_ids
            }
            if not candidate_node_ids:
                return []

        scores: dict[str, float] = {}

        # 1. Semantic vector similarity first
        search_kwargs: dict[str, Any] = {
            "top_k": max(top_k * 2, 10),
            "allowed_node_ids": candidate_node_ids,
            "api_key": api_key,
        }
        if query_vector is not None:
            search_kwargs["query_vector"] = query_vector
        vector_hits = self._vector_store.search(query, **search_kwargs)
        for nid, sim in vector_hits:
            if nid not in scores or sim > scores[nid]:
                scores[nid] = sim

        # 2. Exact full-query alias match (score 1.0)
        normalized_query = _normalize_term(query)
        if normalized_query:
            for nid in self._alias_index.get(normalized_query, ()):
                if candidate_node_ids is None or nid in candidate_node_ids:
                    scores[nid] = max(scores.get(nid, 0.0), 1.0)

            # 3. Multi-word phrase alias match
            for term, nids in self._alias_index.items():
                words = term.split()
                if len(words) < 2:
                    continue
                if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized_query):
                    phrase_score = min(0.95, 0.78 + 0.04 * len(words))
                    for nid in nids:
                        if candidate_node_ids is None or nid in candidate_node_ids:
                            scores[nid] = max(scores.get(nid, 0.0), phrase_score)

        # 4. Fallback 1-word token alias lookup only if confident hits < top_k
        if sum(1 for s in scores.values() if s >= 0.45) < top_k:
            for token in set(_TOKEN_RE.findall(query.lower())):
                for nid in self._alias_index.get(token, ()):
                    if candidate_node_ids is None or nid in candidate_node_ids:
                        scores[nid] = max(scores.get(nid, 0.0), 0.50)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_k]

    def traverse_k_hops(
        self,
        seed_node_ids: list[str],
        k_hops: int = 2,
        allowed_doc_ids: set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], set[str]]:
        """BFS up to ``k_hops`` over incoming and outgoing edges filtered by ``allowed_doc_ids``.

        Returns ``(traversed_edges_list, matched_doc_ids)``.
        """
        visited_nodes: set[str] = set()
        visited_edges: set[tuple[str, str]] = set()
        traversed_edges: list[dict[str, Any]] = []
        matched_doc_ids: set[str] = set()
        queue: deque[tuple[str, int]] = deque()

        for seed_id in seed_node_ids:
            node = self.get_node(seed_id)
            if node is None:
                continue
            if allowed_doc_ids is not None and not (node.doc_ids & allowed_doc_ids):
                continue
            if seed_id not in visited_nodes:
                visited_nodes.add(seed_id)
                queue.append((seed_id, 0))
                matched_doc_ids.update(
                    node.doc_ids & allowed_doc_ids if allowed_doc_ids is not None else node.doc_ids
                )

        while queue:
            current_id, depth = queue.popleft()
            if depth >= k_hops:
                continue

            incident = list(self._graph.out_edges(current_id, data=True)) + list(
                self._graph.in_edges(current_id, data=True)
            )
            for u, v, data in incident:
                edge: GraphEdge = data["edge"]
                edge_docs = (
                    edge.doc_ids & allowed_doc_ids
                    if allowed_doc_ids is not None
                    else edge.doc_ids
                )
                if not edge_docs:
                    continue

                neighbor_id = v if u == current_id else u
                neighbor_node = self.get_node(neighbor_id)
                if neighbor_node is None:
                    continue
                neighbor_docs = (
                    neighbor_node.doc_ids & allowed_doc_ids
                    if allowed_doc_ids is not None
                    else neighbor_node.doc_ids
                )
                if not neighbor_docs:
                    continue

                edge_key = (u, v)
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    filtered_evidence = [
                        e
                        for e in edge.evidence
                        if allowed_doc_ids is None or e.get("doc_id") in allowed_doc_ids
                    ]
                    traversed_edges.append(
                        {
                            "source": edge.source,
                            "target": edge.target,
                            "relation": edge.relation,
                            "doc_ids": sorted(edge_docs),
                            "evidence": filtered_evidence,
                            "hop": depth + 1,
                        }
                    )
                    matched_doc_ids.update(edge_docs)

                if neighbor_id not in visited_nodes:
                    visited_nodes.add(neighbor_id)
                    matched_doc_ids.update(neighbor_docs)
                    queue.append((neighbor_id, depth + 1))

        return traversed_edges, matched_doc_ids

    def save(self, directory: Path) -> None:
        """Serialize all graph nodes and edges to ``graph.json`` and persist vectors."""
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": [
                data["node"].model_dump(mode="json")
                for _, data in self._graph.nodes(data=True)
            ],
            "edges": [
                data["edge"].model_dump(mode="json")
                for _, _, data in self._graph.edges(data=True)
            ],
        }
        (target_dir / "graph.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._vector_store.save(target_dir)

    def load(self, directory: Path) -> None:
        """Restore DiGraph nodes/edges, rebuild alias index, and load vectors from ``directory``."""
        target_dir = Path(directory)
        graph_path = target_dir / "graph.json"
        if graph_path.is_file():
            payload = json.loads(graph_path.read_text(encoding="utf-8"))
            self._graph.clear()
            self._alias_index.clear()
            for node_data in payload.get("nodes", []):
                node = GraphNode.model_validate(node_data)
                self._graph.add_node(node.node_id, node=node)
                self._index_node_aliases(node)
            for edge_data in payload.get("edges", []):
                edge = GraphEdge.model_validate(edge_data)
                self._graph.add_edge(edge.source, edge.target, edge=edge)

        self._vector_store.load(target_dir)

    def _index_node_aliases(self, node: GraphNode) -> None:
        terms = [node.canonical_name, *node.aliases]
        for term in terms:
            norm = _normalize_term(term)
            if norm:
                self._alias_index.setdefault(norm, set()).add(node.node_id)

    def _unindex_node_aliases(self, node: GraphNode) -> None:
        terms = [node.canonical_name, *node.aliases]
        for term in terms:
            norm = _normalize_term(term)
            if not norm:
                continue
            bucket = self._alias_index.get(norm)
            if bucket is not None:
                bucket.discard(node.node_id)
                if not bucket:
                    del self._alias_index[norm]


def _merge_aliases(existing: list[str], incoming: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for item in [*existing, *incoming]:
        norm = _normalize_term(item)
        if norm and norm not in seen:
            seen.add(norm)
            merged.append(item.strip())
    return merged
