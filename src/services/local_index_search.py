import os
import re
import json
import numpy as np

# Drop-in replacement for BraveSearchService / WikipediaSearchService that serves
# results from a *local* corpus index (built by scripts/build_frames_index.py).
#
# Why: live web/Wikipedia search returns short, keyword-highlighted snippets that
# can confuse the agent into long redundant search loops. A local index over the
# task-relevant Wikipedia pages (+ distractors) returns full, substantive passages
# while holding retrieval quality fixed across phrasings — exactly what the
# behavioral experiments need.
#
# Two interchangeable backends (toggle via ``backend=``):
#   - "bm25":  sparse lexical retrieval (rank_bm25), CPU-cheap, built in-memory.
#   - "dense": semantic retrieval (sentence-transformers + FAISS) over precomputed
#              passage embeddings (embeddings.npy produced by the build script).
#
# Both expose ``search(query, max_results) -> [{"title", "snippet"}, ...]`` so the
# service is a true drop-in anywhere a search tool is expected.

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Reasonable small/fast default; must match what the build script used for dense.
DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _tokenize(text: str):
    return _TOKEN_RE.findall((text or "").lower())


class LocalIndexSearchService:
    """Search a local passage corpus by BM25 or dense embeddings."""

    def __init__(self, index_dir: str, backend: str = "bm25",
                 dense_model: str = DEFAULT_DENSE_MODEL):
        if backend not in ("bm25", "dense"):
            raise ValueError(f"backend must be 'bm25' or 'dense', got {backend!r}")
        self.index_dir = index_dir
        self.backend = backend
        self.dense_model_name = dense_model

        corpus_path = os.path.join(index_dir, "corpus.jsonl")
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(
                f"Corpus not found at {corpus_path}. Build it with "
                f"scripts/build_frames_index.py first."
            )

        # Load passages: each row is {"doc_id", "title", "passage_id", "text"}.
        self.passages = []
        with open(corpus_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.passages.append(json.loads(line))
        if not self.passages:
            raise ValueError(f"Corpus at {corpus_path} is empty.")

        if backend == "bm25":
            self._init_bm25()
        else:
            self._init_dense()

    # ---- BM25 backend -----------------------------------------------------
    def _init_bm25(self):
        from rank_bm25 import BM25Okapi
        tokenized = [_tokenize(p["text"]) for p in self.passages]
        self._bm25 = BM25Okapi(tokenized)

    def _bm25_search(self, query: str, max_results: int):
        scores = self._bm25.get_scores(_tokenize(query))
        top = np.argsort(scores)[::-1][:max_results]
        return [int(i) for i in top if scores[i] > 0]

    # ---- Dense backend ----------------------------------------------------
    def _init_dense(self):
        import faiss
        from sentence_transformers import SentenceTransformer

        emb_path = os.path.join(self.index_dir, "embeddings.npy")
        if not os.path.exists(emb_path):
            raise FileNotFoundError(
                f"Dense embeddings not found at {emb_path}. Rebuild with "
                f"`--backend dense` (or `both`)."
            )

        # Prefer the model + query prefix recorded by the build script, so queries
        # are encoded with the SAME model/convention the passages were embedded with.
        # Falls back to the constructor arg (and infers an e5 prefix from the name)
        # for indexes built before the manifest recorded these fields.
        manifest = {}
        manifest_path = os.path.join(self.index_dir, "manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        model_name = manifest.get("dense_model") or self.dense_model_name
        if "query_prefix" in manifest:
            self._query_prefix = manifest["query_prefix"]
        else:
            self._query_prefix = "query: " if "e5" in model_name.lower() else ""

        embeddings = np.load(emb_path).astype("float32")
        if embeddings.shape[0] != len(self.passages):
            raise ValueError(
                f"embeddings rows ({embeddings.shape[0]}) != passages "
                f"({len(self.passages)}); rebuild the dense index."
            )

        self._model = SentenceTransformer(model_name)
        model_dim = self._model.get_sentence_embedding_dimension()
        if model_dim != embeddings.shape[1]:
            raise ValueError(
                f"Dense model '{model_name}' produces {model_dim}-dim vectors but "
                f"embeddings.npy is {embeddings.shape[1]}-dim. The query model must "
                f"match the model used to build the index. Pass the correct "
                f"--dense-model, or rebuild the index with build_frames_index.py."
            )

        faiss.normalize_L2(embeddings)
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

    def _dense_search(self, query: str, max_results: int):
        import faiss
        q = self._model.encode([self._query_prefix + query]).astype("float32")
        faiss.normalize_L2(q)
        _, idx = self._index.search(q, max_results)
        return [int(i) for i in idx[0] if i >= 0]

    # ---- Public interface (matches BraveSearchService) --------------------
    def search(self, query: str, max_results: int = 10):
        """
        Search the local FRAMES corpus and return the most relevant passages.

        Args:
            query: The search query string
            max_results: Maximum number of passages to retrieve

        Returns:
            A list of {"title": str, "snippet": str} dicts, ranked by relevance,
            where each snippet is a full passage from a Wikipedia article.
        """
        if self.backend == "bm25":
            hits = self._bm25_search(query, max_results)
        else:
            hits = self._dense_search(query, max_results)
        return [
            {"title": self.passages[i]["title"], "snippet": self.passages[i]["text"]}
            for i in hits
        ]


if __name__ == "__main__":
    import sys
    index_dir = sys.argv[1] if len(sys.argv) > 1 else "data/frames_index"
    svc = LocalIndexSearchService(index_dir, backend="bm25")
    print(f"Loaded {len(svc.passages)} passages from {index_dir}")
    for r in svc.search("15th First Lady of the United States mother", 3):
        print(f"- {r['title']}: {r['snippet'][:160]}...")
