# Retrieval and vector audit

Prepared by Codex as an assisted audit. The student must review these judgments;
this file is not an individual reflection or a record of student approval.

## Task 1: prediction and manual check

Before the first Q1 run, Codex recorded A1 and A2 as the required evidence in the
working chat. A1 establishes the general access hours; A2 establishes the
two-person rule while the laser operates. This prediction was made before
observing retrieval output; no student prediction has been claimed.

All six paragraphs match the handout verbatim. Their IDs and filenames are
metadata. The source verification test checks all six texts against a separate
transcription. Each file contains two paragraphs, giving six chunks total.

For Q1, the distinct remaining query tokens are `trained`, `graduate`, `student`,
`operate`, `laser`, `alone`, `19`, and `00`. A1 shares exactly `trained`,
`graduate`, and `00`, so its score is **3**. The time strings split on the colon;
`19:00` and `08:00` share `00`. This is a literal word match, not time reasoning.
`student` does not match `students`; `operate` does not match `operating`.
Repeating `laser` in the question cannot increase a score because a set holds
each word once. Q1 selects A1 (3) then A2 (2), so both required passages are present.

## Task 2: interpreting the actual model output

The document matrix is **(6, 384)**: one row per passage and 384 coordinates per
row. Each question vector is **(384,)**. The question and passage coordinates
must have the same length and come from a compatible model space: a dot product
pairs corresponding coordinates, and matching dimensions alone would not make
two unrelated models compatible. The same pretrained model encodes both roles.
This project does not train or fine-tune it.

`encode_document` and `encode_query` distinguish the roles. This MiniLM model
uses the same underlying encoding behavior for them. Neither operation writes
an answer. All 384 coordinates contribute to retrieval; the first 12 are only
an inspection window. Sorting six full-precision scores in descending order,
breaking equal scores by ID, then slicing the first k entries selects passages.

Among the first 12 displayed document coordinates and Q1 coordinates, the largest
is T1 coordinate 12, **0.1357**, and the smallest is T1 coordinate 5, **-0.1167**
(one-based positions). Neither value alone labels what T1 means. The model
represents relationships across the whole vector; individual coordinates do not
have reliable semantic names.

All ten vectors (six documents and four queries) have length approximately 1,
within absolute tolerance 1e-6. The Q1–A2 first-12-product subtotal is
**0.016111500360** and the remaining-372 subtotal is **0.670334187403**.
Their sum is **0.686445687763**. Direct cosine is **0.686445636146**;
the tiny difference comes from finite-precision normalization. All six scores
for every query pass the same cosine agreement check. Model float32 outputs are
preserved in `vectors.npz`; arithmetic is performed in float64, without first
rounding the displayed coordinates. Printed subtotals can differ in the last
decimal when added because display values are rounded.

## Task 3: evidence judgments at k=2

| Question | Keyword selection | Embedding selection | Required evidence coverage |
|---|---|---|---|
| Q1 | A1 (3), A2 (2) | A2 (0.686446), T1 (0.581953) | Keyword 2/2; embeddings 1/2 |
| Q2 | A2 (2), B1 (1) | A2 (0.585193), T1 (0.434761) | Both 1/1 |
| Q3 | T1 (2), A2 (1) | T1 (0.742027), B1 (0.471195) | Both 1/1 |
| Q4 | A2 (1), B1 (1) | T1 (0.347721), A2 (0.320827) | Not applicable: answer absent |

Q1 is a documented embedding retrieval failure under the handout's complete-
evidence criterion. A2 supports the no-alone operating conclusion, but A1 is
absent, so the retrieved context cannot establish whether 19:00 is within access
hours. T1 adds a current-training condition, not the missing access rule.

Q2's paraphrase about another person retrieves A2 at 0.585193. For Q3,
`refresh my laser qualification` retrieves T1 at 0.742027 despite using different
words for training renewal. Keyword retrieval also finds T1 because of `must`
and `laser`; this particular question therefore does not establish an embedding
advantage over the baseline. The extra B1 booking passage does not establish
training-renewal frequency.

Q4 has no answer in any of the six sources. The embedding algorithm always takes
the nearest k passages without an answerability threshold. A positive score
measures alignment in the model's representation, not the presence of a wavelength
fact. Neither method succeeds at retrieving an answer that does not exist. The
CSV uses `not_applicable` for its completeness fields; the observation records
the necessary missing-information check.

The keyword counts and embedding similarities use different scales and must not
be compared numerically to each other. Coverage measures required evidence in
the selected set, not whether every selected passage is relevant. These three
answerable questions are a small classroom diagnostic, not a reliable performance
estimate.

## Changing k for Q1

| k | Selected IDs | Required evidence coverage | What changed |
|---|---|---|---|
| 1 | A2 | 1/2 | Gives the operating rule only |
| 2 | A2; T1 | 1/2 | Adds the current-record and 12-month renewal conditions |
| 3 | A2; T1; B1 | 1/2 | Adds booking duration and calendar procedure |

A1 ranks fourth (0.374249), so neither requested increase completes the evidence.
The extra material is related to using the laser but does not supply the missing
access-hours rule. Increasing k is not guaranteed to fix a retrieval failure.
