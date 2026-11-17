"""Local vector store, deterministic n-gram hash embedder, and LiteLLM embedder."""

from collections.abc import Iterable
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Protocol, runtime_checkable

import numpy as np
import litellm

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    """Protocol for text embedding providers."""

    def embed(self, text: str, api_key: str | None = None) -> np.ndarray:
        """Return a 1-D float32 embedding vector for ``text``."""
        ...

    def embed_many(self, texts: list[str], api_key: str | None = None) -> list[np.ndarray]:
        """Return 1-D float32 embedding vectors for each item in ``texts``."""
        ...


class DeterministicHashEmbedder:
    """Zero-dependency character/word n-gram hashing embedder.

    Produces deterministic, L2-normalized ``float32`` vectors without external
    API keys or model downloads.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.dim = dim

    def embed(self, text: str, api_key: str | None = None) -> np.ndarray:
        """Embed ``text`` into an L2-normalized ``float32`` vector of length ``dim``."""
        vec = np.zeros(self.dim, dtype=np.float32)
        normalized = " ".join(text.lower().split())
        if not normalized:
            return vec

        tokens = _TOKEN_RE.findall(normalized)
        for token in tokens:
            self._accumulate(vec, f"w:{token}", weight=2.0)
            if len(token) >= 3:
                for i in range(len(token) - 2):
                    self._accumulate(vec, f"c3:{token[i : i + 3]}", weight=1.0)
            if len(token) >= 4:
                for i in range(len(token) - 3):
                    self._accumulate(vec, f"c4:{token[i : i + 4]}", weight=1.0)

        for i in range(len(tokens) - 1):
            self._accumulate(vec, f"b:{tokens[i]}_{tokens[i + 1]}", weight=1.5)

        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def embed_many(self, texts: list[str], api_key: str | None = None) -> list[np.ndarray]:
        """Embed a batch of ``texts`` into L2-normalized ``float32`` vectors."""
        return [self.embed(text, api_key=api_key) for text in texts]

    def _accumulate(self, vec: np.ndarray, feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, byteorder="little", signed=False)
        idx = raw % self.dim
        sign = 1.0 if ((raw >> 63) & 1) == 0 else -1.0
        vec[idx] += sign * weight


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0.0:
        return arr / norm
    return arr


class LiteLLMEmbedder:
    """Embedding provider backed by ``litellm.embedding``."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def embed(self, text: str, api_key: str | None = None) -> np.ndarray:
        """Embed ``text`` via ``litellm.embedding`` and return an L2-normalized ``float32`` vector."""
        return self.embed_many([text], api_key=api_key)[0]

    def embed_many(self, texts: list[str], api_key: str | None = None) -> list[np.ndarray]:
        """Embed ``texts`` in chunks of 100 via ``litellm.embedding``."""
        if not texts:
            return []

        resolved_key = api_key or self.api_key
        vectors: list[np.ndarray] = []
        batch_size = 100
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            kwargs: dict[str, Any] = {
                "model": self.model,
                "input": batch,
                "timeout": self.timeout,
            }
            if resolved_key:
                kwargs["api_key"] = resolved_key

            response = litellm.embedding(**kwargs)
            items = list(response.data)
            if items and all(
                (isinstance(it, dict) and "index" in it) or hasattr(it, "index")
                for it in items
            ):
                items.sort(key=lambda it: it["index"] if isinstance(it, dict) else it.index)

            for item in items:
                raw_embedding = item["embedding"] if isinstance(item, dict) else item.embedding
                vectors.append(_l2_normalize(np.asarray(raw_embedding, dtype=np.float32)))

        return vectors


class LocalVectorStore:
    """In-memory cosine-similarity vector store keyed by ``node_id``."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self._embedder: Embedder = embedder or DeterministicHashEmbedder()
        self._vectors: dict[str, np.ndarray] = {}
        self._texts: dict[str, str] = {}
        self._entry_types: dict[str, str] = {}
        self._doc_ids: dict[str, str | None] = {}

    def _embed_texts(self, texts: list[str], api_key: str | None = None) -> list[np.ndarray]:
        embed_many_fn = getattr(self._embedder, "embed_many", None)
        if callable(embed_many_fn):
            return embed_many_fn(texts, api_key=api_key)
        return [self._embedder.embed(text, api_key=api_key) for text in texts]

    def _reconcile_dimension(
        self,
        target_shape: tuple[int, ...],
        api_key: str | None = None,
    ) -> None:
        """Re-embed or prune stored vectors whose shape differs from ``target_shape``."""
        mismatched = [nid for nid, vec in self._vectors.items() if vec.shape != target_shape]
        if not mismatched:
            return

        reembed_ids = [nid for nid in mismatched if nid in self._texts]
        for nid in mismatched:
            if nid not in self._texts:
                self._vectors.pop(nid, None)
                self._entry_types.pop(nid, None)
                self._doc_ids.pop(nid, None)

        if reembed_ids:
            vectors = self._embed_texts([self._texts[nid] for nid in reembed_ids], api_key=api_key)
            for nid, vec in zip(reembed_ids, vectors, strict=True):
                normalized = _l2_normalize(vec)
                if normalized.shape != target_shape:
                    raise ValueError(
                        f"Re-embedded vector for {nid!r} has shape {normalized.shape}, expected {target_shape}"
                    )
                self._vectors[nid] = normalized

    def upsert(
        self,
        node_id: str,
        text: str,
        api_key: str | None = None,
        entry_type: str = "node",
        doc_id: str | None = None,
    ) -> None:
        """Embed and store ``text`` under ``node_id``."""
        vec = _l2_normalize(self._embedder.embed(text, api_key=api_key))
        self._vectors[node_id] = vec
        self._texts[node_id] = text
        self._entry_types[node_id] = entry_type
        self._doc_ids[node_id] = doc_id
        self._reconcile_dimension(vec.shape, api_key=api_key)

    def upsert_many(
        self,
        items: list[tuple[str, str]],
        api_key: str | None = None,
        entry_type: str = "node",
        doc_id: str | None = None,
    ) -> None:
        """Embed and store multiple ``(node_id, text)`` pairs in batch."""
        if not items:
            return
        texts = [text for _, text in items]
        vectors = self._embed_texts(texts, api_key=api_key)
        for (node_id, text), vec in zip(items, vectors, strict=True):
            self._vectors[node_id] = _l2_normalize(vec)
            self._texts[node_id] = text
            self._entry_types[node_id] = entry_type
            self._doc_ids[node_id] = doc_id
        self._reconcile_dimension(self._vectors[items[0][0]].shape, api_key=api_key)

    def delete(self, node_id: str) -> bool:
        """Remove ``node_id`` from the store. Returns ``True`` if it existed."""
        existed = node_id in self._vectors
        self._vectors.pop(node_id, None)
        self._texts.pop(node_id, None)
        self._entry_types.pop(node_id, None)
        self._doc_ids.pop(node_id, None)
        return existed

    def delete_many(self, node_ids: Iterable[str]) -> int:
        """Remove multiple ``node_ids`` and return the count of removed entries."""
        removed = 0
        for node_id in node_ids:
            if self.delete(node_id):
                removed += 1
        return removed

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Remove all ``chunk`` and ``table`` entries associated with ``doc_id``."""
        matching_ids = [
            nid
            for nid in list(self._vectors.keys())
            if self._doc_ids.get(nid) == doc_id
            and self._entry_types.get(nid) in {"chunk", "table"}
        ]
        return self.delete_many(matching_ids)

    def search(
        self,
        query: str,
        top_k: int = 5,
        allowed_node_ids: set[str] | None = None,
        api_key: str | None = None,
        entry_types: set[str] | None = None,
        allowed_doc_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return up to ``top_k`` ``(node_id, cosine_similarity)`` pairs descending."""
        if top_k <= 0 or not self._vectors:
            return []

        query_vec = _l2_normalize(self._embedder.embed(query, api_key=api_key))
        if float(np.linalg.norm(query_vec)) == 0.0:
            return []

        self._reconcile_dimension(query_vec.shape, api_key=api_key)

        candidate_ids: list[str] = []
        for nid in self._vectors:
            etype = self._entry_types.get(nid, "node")
            if entry_types is not None and etype not in entry_types:
                continue
            if etype == "node":
                if allowed_node_ids is not None and nid not in allowed_node_ids:
                    continue
            else:
                if allowed_doc_ids is not None and self._doc_ids.get(nid) not in allowed_doc_ids:
                    continue
            candidate_ids.append(nid)
        if not candidate_ids:
            return []

        scored: list[tuple[str, float]] = [
            (nid, float(np.dot(query_vec, self._vectors[nid])))
            for nid in candidate_ids
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]

    def count(self) -> int:
        """Return the number of stored node embeddings."""
        return len(self._vectors)

    def save(self, directory: Path) -> None:
        """Persist stacked float32 embeddings to ``vectors.npz`` and texts to ``vector_texts.json``."""
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        if self._vectors:
            node_ids = np.array(list(self._vectors.keys()), dtype=str)
            arrays = [np.asarray(self._vectors[nid], dtype=np.float32) for nid in node_ids]
            shapes = {arr.shape for arr in arrays}
            if len(shapes) > 1:
                raise ValueError(
                    f"Vector dimension mismatch in store: found shapes {sorted(shapes)}"
                )
            matrix = np.stack(arrays).astype(np.float32)
        else:
            node_ids = np.array([], dtype=str)
            matrix = np.empty((0, 0), dtype=np.float32)

        np.savez(target_dir / "vectors.npz", node_ids=node_ids, matrix=matrix)
        (target_dir / "vector_texts.json").write_text(
            json.dumps(self._texts, indent=2),
            encoding="utf-8",
        )
        (target_dir / "vector_meta.json").write_text(
            json.dumps(
                {
                    "entry_types": self._entry_types,
                    "doc_ids": self._doc_ids,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self, directory: Path) -> None:
        """Restore ``_vectors``, ``_texts``, ``_entry_types``, and ``_doc_ids`` from ``directory``."""
        target_dir = Path(directory)
        npz_path = target_dir / "vectors.npz"
        texts_path = target_dir / "vector_texts.json"
        meta_path = target_dir / "vector_meta.json"

        if npz_path.is_file():
            with np.load(npz_path, allow_pickle=False) as data:
                node_ids = data["node_ids"]
                matrix = data["matrix"]
                self._vectors = {
                    str(nid): np.asarray(matrix[idx], dtype=np.float32)
                    for idx, nid in enumerate(node_ids)
                }

        if texts_path.is_file():
            raw_texts = json.loads(texts_path.read_text(encoding="utf-8"))
            self._texts = {str(k): str(v) for k, v in raw_texts.items()}

        if meta_path.is_file():
            raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._entry_types = {
                str(k): str(v) for k, v in raw_meta.get("entry_types", {}).items()
            }
            self._doc_ids = {
                str(k): (str(v) if v is not None else None)
                for k, v in raw_meta.get("doc_ids", {}).items()
            }
        else:
            self._entry_types = {nid: "node" for nid in self._vectors}
            self._doc_ids = {nid: None for nid in self._vectors}
