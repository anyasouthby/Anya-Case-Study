import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from chunking import create_chunks


MODEL_NAME = "all-MiniLM-L6-v2"

CACHE_DIR = Path(".cache")
CACHE_FILE = CACHE_DIR / "retrieval_index.npz"
MANIFEST_FILE = CACHE_DIR / "retrieval_manifest.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


class Retriever:
    """Semantic search over the document collection."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.model = SentenceTransformer(MODEL_NAME)

        cache_key = self._create_cache_key()

        if self._cache_is_valid(cache_key):
            print("Loading retrieval index from cache...")
            self._load_cache()
        else:
            print("Building retrieval index...")
            self._build_index(cache_key)

    def _create_cache_key(self) -> str:
        """Create a fingerprint representing the current dataset and settings."""

        hasher = hashlib.sha256()

        for file_path in sorted(self.dataset_path.iterdir()):
            if file_path.is_file() and not file_path.name.startswith("."):
                hasher.update(file_path.name.encode("utf-8"))
                hasher.update(file_path.read_bytes())

        hasher.update(MODEL_NAME.encode("utf-8"))
        hasher.update(str(CHUNK_SIZE).encode("utf-8"))
        hasher.update(str(CHUNK_OVERLAP).encode("utf-8"))

        return hasher.hexdigest()

    def _cache_is_valid(self, cache_key: str) -> bool:
        """Check whether a usable cache exists for the current dataset."""

        if not CACHE_FILE.exists() or not MANIFEST_FILE.exists():
            return False

        try:
            manifest = json.loads(
                MANIFEST_FILE.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            return False

        return manifest.get("cache_key") == cache_key

    def _build_index(self, cache_key: str):
        """Create chunks and embeddings, then save them to disk."""

        self.chunks = create_chunks(
            str(self.dataset_path)
        )

        print(f"Loaded {len(self.chunks)} chunks.")

        embeddings = self.model.encode(
            [chunk["text"] for chunk in self.chunks],
            show_progress_bar=True,
        )

        self.embeddings = np.asarray(embeddings)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        np.savez(
            CACHE_FILE,
            embeddings=self.embeddings,
        )

        MANIFEST_FILE.write_text(
            json.dumps(
                {
                    "cache_key": cache_key,
                    "model_name": MODEL_NAME,
                    "chunk_size": CHUNK_SIZE,
                    "chunk_overlap": CHUNK_OVERLAP,
                    "num_chunks": len(self.chunks),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # Store the chunks separately so they do not need to be
        # regenerated from the source documents on every startup.
        chunks_file = CACHE_DIR / "chunks.json"

        chunks_file.write_text(
            json.dumps(self.chunks, ensure_ascii=False),
            encoding="utf-8",
        )

        print("Retrieval index saved to cache.")

    def _load_cache(self):
        """Load previously generated chunks and embeddings from disk."""

        chunks_file = CACHE_DIR / "chunks.json"

        self.embeddings = np.load(
            CACHE_FILE
        )["embeddings"]

        self.chunks = json.loads(
            chunks_file.read_text(encoding="utf-8")
        )

        print(f"Loaded {len(self.chunks)} chunks from cache.")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the most relevant document chunks for a query."""

        query_embedding = self.model.encode([query])

        scores = self.model.similarity(
            query_embedding,
            self.embeddings,
        )[0]

        ranked_indices = scores.argsort(descending=True)[:top_k]

        return [
            self.chunks[int(index)]
            for index in ranked_indices
        ]


if __name__ == "__main__":
    retriever = Retriever("data/dataset")

    query = "What are the main risks associated with AI?"

    results = retriever.search(query)

    print(f"\nQuery: {query}")
    print(f"Found {len(results)} results.")

    for result in results:
        print("\n" + "=" * 80)
        print(
            f"{result['filename']} | "
            f"chunk {result['chunk_id']}"
        )
        print("=" * 80)
        print(result["text"][:500])

