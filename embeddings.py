"""Real CPU embeddings, visible vectors, and explicit similarity arithmetic."""
import argparse
from importlib.metadata import version
import platform
import numpy as np
from sentence_transformers import SentenceTransformer
from baseline import ROOT, load_chunks

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class EmbeddingIndex:
    def __init__(self, chunks):
        self.chunks = chunks
        self.revision = REVISION
        self.model = SentenceTransformer(MODEL, revision=self.revision, device="cpu",
                                         cache_folder=str(ROOT / ".cache"))
        # Text only: labels and filenames never enter the model.
        self.documents = self.model.encode_document(
            [c.text for c in chunks], normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False)
        assert self.documents.shape == (6, 384), self.documents.shape
        assert np.allclose(np.linalg.norm(self.documents, axis=1), 1, atol=1e-6, rtol=0)

    def query(self, question, k=2):
        if k < 1:
            raise ValueError("k must be positive")
        vector = self.model.encode_query(
            question, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False)
        assert vector.shape == (384,), vector.shape
        assert np.isclose(np.linalg.norm(vector), 1, atol=1e-6, rtol=0)
        # Use all coordinates, with float64 arithmetic on the model's float32 outputs.
        scores = self.documents.astype(np.float64) @ vector.astype(np.float64)
        order = sorted(range(len(self.chunks)), key=lambda i: (-scores[i], self.chunks[i].id))
        return vector, scores, order, order[:k]

    def environment(self):
        print("Model: " + MODEL)
        print("Device: " + str(self.model.device))
        print("Model revision: " + self.revision)
        print("Python: " + platform.python_version())
        for name in ("sentence-transformers", "numpy", "torch", "transformers", "huggingface-hub"):
            print(f"{name}: {version(name)}")
        print(f"Document matrix shape: {self.documents.shape}; dtype: {self.documents.dtype}")
        print("All 384 coordinates are used; only the first 12 are displayed.")
        for chunk, vector in zip(self.chunks, self.documents):
            print(f"{chunk.id} first 12: " + " ".join(f"{v:.4f}" for v in vector[:12]))
            print(f"{chunk.id} length={np.linalg.norm(vector):.9f}; approximately 1: True")

    def show(self, question, result, products=False):
        vector, scores, order, selected = result
        print(f"Question: {question}")
        print(f"Query shape: {vector.shape}; dtype: {vector.dtype}")
        print("Query first 12: " + " ".join(f"{v:.4f}" for v in vector[:12]))
        print(f"Query length={np.linalg.norm(vector):.9f}; approximately 1: True")
        print("Embedding scores (all six):")
        for i in order:
            print(f"{self.chunks[i].id} score={scores[i]:.9f}")
        print("Selected IDs: " + ";".join(self.chunks[i].id for i in selected))
        for i in selected:
            c = self.chunks[i]
            print(f"[{c.id}] {c.filename}: {c.text}")
        d = self.documents.astype(np.float64)
        q = vector.astype(np.float64)
        direct_cosines = (d @ q) / (np.linalg.norm(d, axis=1) * np.linalg.norm(q))
        agree = np.allclose(scores, direct_cosines, atol=1e-6, rtol=0)
        print(f"All six dot products agree with direct cosine (atol=1e-6): {agree}")
        assert agree
        if products:
            top = order[0]
            multiplied = d[top] * q
            print(f"Coordinate products for top passage {self.chunks[top].id}:")
            print(" ".join(f"{v:.9f}" for v in multiplied[:12]))
            print(f"First 12 subtotal: {multiplied[:12].sum():.12f}")
            print(f"Remaining 372 subtotal: {multiplied[12:].sum():.12f}")
            print(f"Full dot product: {scores[top]:.12f}")
            print(f"Direct cosine: {direct_cosines[top]:.12f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=2)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("k must be positive")
    index = EmbeddingIndex(load_chunks())
    index.environment()
    index.show(args.question, index.query(args.question, args.k), products=True)
