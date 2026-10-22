"""Transparent word-overlap retrieval; needs only Python's standard library."""
import argparse
from dataclasses import dataclass
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
STOPWORDS = set("a an the is are to of in at for and or may can i it be".split())


@dataclass(frozen=True)
class Chunk:
    id: str
    filename: str
    text: str


def load_chunks(directory=ROOT / "docs"):
    chunks = []
    for path in sorted(Path(directory).glob("*.txt")):
        for paragraph in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip()):
            match = re.fullmatch(r"\[([A-Z]\d+)\] (.+)", paragraph)
            if not match:
                raise ValueError(f"Invalid labeled paragraph in {path.name}: {paragraph!r}")
            chunks.append(Chunk(match[1], path.name, match[2]))
    if len(chunks) != 6 or {c.id for c in chunks} != {"A1", "A2", "B1", "B2", "T1", "T2"}:
        raise ValueError("Expected exactly six uniquely labeled Cedar chunks")
    return sorted(chunks, key=lambda c: c.id)


def words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOPWORDS


def retrieve(chunks, question, k=2):
    if k < 1:
        raise ValueError("k must be positive")
    query_words = words(question)
    ranked = [(c, len(query_words & words(c.text)), sorted(query_words & words(c.text)))
              for c in chunks]
    ranked.sort(key=lambda row: (-row[1], row[0].id))
    selected = [row for row in ranked if row[1] > 0][:k]
    return ranked, selected


def show(question, ranked, selected):
    print(f"Question: {question}")
    print("Keyword scores (all six; distinct shared words):")
    for chunk, score, matches in ranked:
        print(f"{chunk.id} score={score} matched={','.join(matches) or '(none)'}")
    if not selected:
        print("No matching passages.")
    else:
        print("Selected IDs: " + ";".join(row[0].id for row in selected))
        for chunk, _, _ in selected:
            print(f"[{chunk.id}] {chunk.filename}: {chunk.text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=2)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("k must be positive")
    ranking, selection = retrieve(load_chunks(), args.question, args.k)
    show(args.question, ranking, selection)
