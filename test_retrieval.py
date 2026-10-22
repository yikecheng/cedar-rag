"""Checks the handout contract and independently verifies the saved arithmetic."""
import csv
import hashlib
import io
import json
from contextlib import redirect_stdout
import unittest
import numpy as np
from baseline import ROOT, STOPWORDS, load_chunks, retrieve, show, words

# Independently transcribed from the handout to detect source drift.
EXPECTED = {
    "A1": ("access.txt", "General laboratory access hours are 08:00 to 22:00 for trained graduate students. Access outside these hours requires written approval from the lab manager."),
    "A2": ("access.txt", "The laser room requires two trained people to be present whenever the laser is operating, including during general laboratory access hours."),
    "T1": ("training.txt", "Laser training must be renewed every 12 months. A current training record is required before operating the laser."),
    "T2": ("training.txt", "Visitors may enter the general laboratory only when accompanied by a trained member of staff."),
    "B1": ("booking.txt", "Laser room bookings are limited to two hours per session. Reserve the room through the equipment calendar."),
    "B2": ("booking.txt", "Cancel an unused booking at least four hours before its start time."),
}
Q1 = "May a trained graduate student operate the laser alone at 19:00?"


class RetrievalContract(unittest.TestCase):
    def test_verbatim_sources_and_metadata(self):
        chunks = load_chunks()
        self.assertEqual(len(chunks), 6)
        self.assertEqual({c.id: (c.filename, c.text) for c in chunks}, EXPECTED)

    def test_exact_token_policy(self):
        self.assertEqual(STOPWORDS, {"a", "an", "the", "is", "are", "to", "of", "in", "at",
                                     "for", "and", "or", "may", "can", "i", "it", "be"})
        self.assertEqual(words("The LASER laser, laser! 19:00"), {"laser", "19", "00"})
        self.assertEqual(words("student students operating operate"),
                         {"student", "students", "operating", "operate"})

    def test_q1_scores_and_ties(self):
        ranked, selected = retrieve(load_chunks(), Q1)
        self.assertEqual([(c.id, score) for c, score, _ in ranked],
                         [("A1", 3), ("A2", 2), ("B1", 1), ("T1", 1), ("T2", 1), ("B2", 0)])
        self.assertEqual([c.id for c, _, _ in selected], ["A1", "A2"])
        self.assertEqual(ranked[0][2], ["00", "graduate", "trained"])

    def test_no_repeat_bonus_or_metadata_matching(self):
        chunks = load_chunks()
        once = retrieve(chunks, "laser")[0]
        repeated = retrieve(chunks, "laser laser laser")[0]
        self.assertEqual(once, repeated)
        self.assertEqual(retrieve(chunks, "A1 txt")[1], [])
        # The filename stem also occurs in the text; it should count once there.
        self.assertEqual(retrieve(chunks, "access.txt")[0][0][1], 1)

    def test_zero_matches_and_k(self):
        ranked, selected = retrieve(load_chunks(), "wavelength")
        self.assertEqual(selected, [])
        output = io.StringIO()
        with redirect_stdout(output):
            show("wavelength", ranked, selected)
        self.assertIn("No matching passages.", output.getvalue())
        self.assertEqual(len(retrieve(load_chunks(), "laser", 10)[1]), 3)
        with self.assertRaises(ValueError):
            retrieve(load_chunks(), "laser", 0)

    def test_saved_vectors_and_csv(self):
        evidence = ROOT / "evidence"
        metadata = json.loads((evidence / "run-metadata.json").read_text())
        for filename, digest in metadata["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / filename).read_bytes()).hexdigest(), digest)
        with np.load(evidence / "vectors.npz", allow_pickle=False) as archive:
            documents = archive["documents"].astype(np.float64)
            self.assertEqual(documents.shape, (6, 384))
            np.testing.assert_allclose(np.linalg.norm(documents, axis=1), 1, atol=1e-6, rtol=0)
            with (ROOT / "results.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r["question_id"] for r in rows], ["Q1", "Q2", "Q3", "Q4"])
            for row in rows:
                query = archive[row["question_id"]].astype(np.float64)
                self.assertEqual(query.shape, (384,))
                self.assertAlmostEqual(float(np.linalg.norm(query)), 1, places=6)
                # An explicit loop independently checks matrix multiplication.
                scores = [sum(float(x) * float(y) for x, y in zip(document, query))
                          for document in documents]
                order = sorted(range(6), key=lambda i: (-scores[i], metadata["chunk_order"][i]))
                ids = [metadata["chunk_order"][i] for i in order[:2]]
                self.assertEqual(row["embedding_top_ids"].split(";"), ids)
                np.testing.assert_allclose([scores[i] for i in order[:2]],
                                           list(map(float, row["embedding_scores"].split(";"))),
                                           atol=1e-12, rtol=0)
                cosine = np.array(scores) / (np.linalg.norm(documents, axis=1) * np.linalg.norm(query))
                np.testing.assert_allclose(scores, cosine, atol=1e-6, rtol=0)
            self.assertEqual(rows[-1]["required_ids"], "absent")


if __name__ == "__main__":
    unittest.main(verbosity=2)
