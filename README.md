# Cedar Lab: visible retrieval and evidence limits

Prepared for **Yike Cheng**, with Codex assistance. No partner was specified;
individual-work status should be confirmed before submission. Cedar Lab is
fictional and these passages are classroom material.

This project follows [PS05 and Lesson 10 — RAG Systems](https://docs.google.com/document/d/1EIb-8b-DZx9Ir_Yk3yfA1D6gAtMta1vtQMmclskoXPg/edit).
It compares exact word overlap with a real pretrained embedding model on the
same six passages, then prepares an evidence-only handoff for answer generation.
There is no deployed application, vector database, or external model API.

## Reproduce the actual run

Run these commands from the `cedar-rag` directory. The recorded run used Python
3.10.6 on macOS ARM64, CPU, sentence-transformers 6.2.0, numpy 2.2.6, torch 2.14.1,
transformers 5.20.2, and huggingface-hub 2.2.0. Python 3.10 is the recorded
environment; wheel availability can differ on other platforms. The installed
dependency snapshot is in `requirements-lock.txt`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python baseline.py "May a trained graduate student operate the laser alone at 19:00?" -k 2
.venv/bin/python embeddings.py "May a trained graduate student operate the laser alone at 19:00?" -k 2
.venv/bin/python experiment.py --out rerun
.venv/bin/python -m unittest discover -v
```

The initial package and model downloads require internet access. No GPU or API
key is required. Embedding commands download the public model into `.cache/`,
which is excluded from Git. The model is pinned to the revision actually loaded:
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` of
`sentence-transformers/all-MiniLM-L6-v2`. Once it is cached, run offline:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python experiment.py --out rerun-offline
```

Use a new output directory for each experiment. The script refuses to overwrite
an existing `results.csv`, protecting judgments. To save a separate single-query
inspection, redirect stdout to a new file:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python embeddings.py "May a trained graduate student operate the laser alone at 19:00?" -k 1 > q1-k1.txt
HF_HUB_OFFLINE=1 .venv/bin/python embeddings.py "May a trained graduate student operate the laser alone at 19:00?" -k 3 > q1-k3.txt
```

All committed numeric evidence was produced by this project's actual local CPU
run. No vectors, rankings, or scores are borrowed from the reference assignment.
The model produces float32 vectors; dot products and the direct-cosine check use
float64 arithmetic on those unrounded outputs. Minor last-digit differences are
possible across library versions or hardware. Rounding is only for display.

## How the code works

`baseline.py` uses only the standard library. Lines 18–28 load six paragraphs and
keep IDs and filenames separate from text. Lines 31–32 lowercase text, split it
into alphanumeric words, remove exactly the handout's stopwords, and use a set
so repeated words count once. Lines 38–40 count shared words. Lines 41–42 rank
by descending score with ascending ID as the tiebreaker, then take at most k
positive matches. All six scores and their matched words are printed. There is
no stemming, synonym expansion, or answer generation. `k` defaults to 2; an
empty positive set prints `No matching passages.`

`embeddings.py` lines 17–24 load the pretrained CPU model and encode all six
texts once using `encode_document`. Lines 29–35 use `encode_query` with the same
model and calculate one dot product per document. Both encoding calls normalize
vectors and return numpy arrays. Lines 36–37 sort and select the top k, without
a relevance threshold. Lines 65–79 compare the dot products with direct cosine
and expose the coordinate products and subtotals for the leading passage.

The six document rows form a `(6, 384)` matrix. A question produces a `(384,)`
vector. Normalization divides by vector length, making lengths approximately 1;
the dot product then agrees with cosine within numerical tolerance. Each score
sums products across all 384 matching coordinates. The first 12 coordinates are
displayed for inspection, not used as a shortened vector. Individual coordinates
have no assigned semantic names. This is a pretrained representation, not model
training on the six paragraphs. Ordinary RAG changes the generator's supplied
context rather than its learned weights.

`experiment.py` reuses one document matrix for all four questions, writes raw
rankings before judgments, records k=1/2/3, and exports only each selected pair
to its answer prompt. The question and passage vectors are saved in
`evidence/vectors.npz`; `run-metadata.json` records passage order, source hashes,
model revision, timestamp, and all scores. These allow independent arithmetic
checks without downloading or running the model again.

## What the runs show

| Question | Keyword top 2 | Embedding top 2 | Required evidence |
|---|---|---|---|
| Q1: alone at 19:00 | A1; A2 | A2; T1 | A1 and A2: keyword complete, embeddings partial |
| Q2: without another person | A2; B1 | A2; T1 | A2: both complete |
| Q3: refresh qualification | T1; A2 | T1; B1 | T1: both complete |
| Q4: wavelength | A2; B1 | T1; A2 | Absent: neither retrieves an answer |

Embeddings do not win every comparison. Q1 required coverage is 1/2 for
embeddings at k=1, 2, and 3; A1 ranks fourth. Added training and booking passages
do not establish access hours. Q4 receives positive similarities despite missing
answer evidence, so the final answer must acknowledge the omission. Similarity
is not truth, permission, or answerability. Raw keyword and embedding score
scales should not be compared numerically.

See `evidence/retrieval-audit.md` for the manual score count, vector explanation,
actual score examples, coverage judgments, and interpretation of the k experiment.
`results.csv` contains explicitly labeled **Codex-assisted judgments awaiting
student review**. `evidence/results-unjudged.csv` preserves the original blank
judgment fields required by prompt 3.

## Evidence and Task 4 status

| Requirement | Artifact and status |
|---|---|
| Verbatim sources | `docs/access.txt`, `docs/training.txt`, `docs/booking.txt`; verified |
| Keyword Q1 | `evidence/keyword-q1.txt`; actual output |
| Visible embeddings | `evidence/embedding-inspection.txt`; actual vectors, lengths, dimensions, products, cosine check |
| Four retrieval comparisons | `evidence/retrieval-runs.txt`, `results.csv`; actual scores and assisted judgments |
| k experiment | `evidence/k-experiment.txt`; actual k=1/2/3 selections and coverage |
| Environment | `evidence/environment.txt`, `requirements-lock.txt`, `evidence/run-metadata.json` |
| Automated verification | `evidence/verification.txt`; six contract and arithmetic tests passed |
| Fresh answer chats | Both completed; originals, one Q1 retry, separate audits, and `evidence/answer-provenance.json` preserved |
| Individual reflection | Must be written by the student, at most 150 words; pending |

Each initial prompt contains only the handout's instruction, its question, and
the actual k=2 embedding passages. The user authorized two separate fresh
projectless chats; neither received the full corpus or this project, and no tool
calls were observed. The original Q1 response applied the two-person rule but
omitted the missing access-hours evidence. That original is preserved in
`evidence/q1-answer.txt`. One revised instruction with identical passages ran as
a follow-up in the same Q1 chat. Its actual response in
`evidence/q1-answer-revised.txt` now acknowledges the missing hours explicitly.
Q4 correctly reported that wavelength is unspecified and needed no revision.
See `evidence/q1-answer-audit.md` and `evidence/q4-answer-audit.md` for claim
checks, and `evidence/answer-provenance.json` for chat and turn IDs. These are
assisted audits, awaiting student review.

The substantive human requirements and final submission steps are in
`HUMAN_STEPS.md`. The technical artifacts address visible embeddings (3 points),
retrieval comparison (3), grounded answers (2), and reproducible instructions
(part of the remaining 2). A full 10/10 cannot be claimed before the student's
own audit review and individual reflection are complete.

## Originality and references

The assignment handout supplies the six paragraphs, questions, scoring policy,
and evidence targets. The implementation and explanations here were written
independently with Codex. [Yinuo Zhao's reference repository](https://github.com/inorrr/cedar-rag)
was consulted for its README's organization and evidence provenance approach;
its implementation and personal reflection were not copied, and its scores
were not substituted for this project's model outputs.

Technical references: the [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
describes the 384-dimensional representation; the [Sentence Transformers semantic
search guide](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
explains shared-space retrieval and query/document encoding roles.
