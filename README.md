# Multimodal Image–Text Search

Bidirectional semantic search over Flickr8k through a shared CLIP embedding
space, plus a BLIP visual-question-answering layer — and a categorized
breakdown of where both of them fail.

> **Status:** Phase 1 of 7 complete (data pipeline, run end to end on the real
> dataset). This README grows with each phase; metrics appear as the phases
> that measure them land.

---

## What this is

Two systems over one dataset.

**Search.** CLIP maps images and text into the same 512-dimensional vector
space, which makes a sentence and a photograph directly comparable. All 8,000
images are encoded once and cached, so a query is a nearest-neighbour lookup
rather than a model run. Two query paths share one index: text→image and
image→image.

**Visual QA.** BLIP-VQA answers natural-language questions about a given image.

**The part worth reading.** Both systems are then measured in a way that does
not average away the interesting information: VQA accuracy broken out by
question type, retrieval recall on a deliberately hard set of near-identical
images alongside the easy set, and a human agreement ceiling derived from the
five independent captions each image carries.

---

## Architecture

```mermaid
flowchart TD
    D[Flickr8k · 8,000 images<br/>5 human captions each] --> L[Loader<br/>ImageRecord]
    L --> S[Frozen split<br/>splits.json]
    L --> A[Caption agreement<br/>human ceiling]

    S --> E[CLIP ViT-B/32<br/>image encoder]
    E --> N[L2-normalised vectors<br/>.npy cache]
    N --> X[(ChromaDB index)]

    Q[Text query] --> T[CLIP text encoder]
    T --> X
    I[Query image] --> E
    X --> R[Ranked results<br/>SearchResult]

    R --> M[Recall@k · MRR<br/>hard negatives · latency]
    A -.human ceiling.-> M

    S --> V[BLIP-VQA]
    QQ[Typed question] --> V
    V --> F[Accuracy per question type<br/>failure analysis]

    style A fill:#f9e79f
    style M fill:#aed6f1
    style F fill:#aed6f1
```

CLIP and BLIP are independent models. They share the dataset, not a pipeline:
CLIP does retrieval, BLIP does question answering, and neither consumes the
other's output.

---

## Tech stack, and why

| Component | Choice | Reasoning |
|---|---|---|
| Language | Python 3.11 | Every wheel in the ML stack exists for it; 3.13+ still has gaps in this dependency set |
| Tensors | PyTorch 2.14 (CPU build) | What the Hugging Face model implementations are written against |
| Models | `transformers` | One loading API for both CLIP and BLIP |
| Shared space | CLIP ViT-B/32 | Smallest standard CLIP. ViT-B/16 quadruples the patch count for a marginal gain that CPU inference cannot afford |
| Visual QA | BLIP-VQA-base | Purpose-built for VQA and light enough for CPU |
| Vector store | ChromaDB, local persist | Embedded, no server to run; teaches the production pattern at a scale where NumPy would also have worked |
| Embedding cache | NumPy `.npy` | Re-encoding 8,000 images per experiment would dominate iteration time |
| API | FastAPI | Typed, self-documenting, loads models once at startup |
| Tests | pytest | Model- and data-dependent tests are marked so CI can skip them |
| Lint/format | ruff | Replaces flake8 + black + isort with one fast tool |
| Container | Docker, multi-stage | Weights baked in so the image runs fully offline |

---

## The human ceiling

*From `results/caption_agreement.json`, over all 8,000 images.*

Each image carries five captions written independently by five people. That
redundancy is usually spent as five times more training text. Measured instead,
it says how ambiguous the task itself is — and therefore how well any model
could possibly do.

| Measure | Value | Reading |
|---|---|---|
| Mean pairwise Jaccard | **0.217** | Two annotators share about a fifth of their content words |
| Mean subject consensus | **0.823** | But they agree on *what the photo is of* far more often |
| All five share a subject | **43.7%** | Unambiguous images — a miss here is the model's fault |
| No majority subject | **5.3%** | ~426 images where the annotators themselves disagree |
| Caption length | 11.8 words | within-image spread 3.3 words |

The gap between 0.217 and 0.823 is the finding. People describe the same
photograph in very different words while agreeing on its subject, which is
exactly why lexical matching is the wrong tool and a shared embedding space is
the right one.

The 5.3% matters for Phase 6: when the model fails on an image whose own
annotators could not agree, that is a property of the data, not a model error —
and this file is the evidence for saying so.

Both numbers are *lower bounds*. The measures here are lexical, so they score
"a wakeboarder" and "a parasurfer" as unrelated:

```
test_00837   mean Jaccard 0.009
  - A man jet boarding .
  - A parasurfer is airborne over the water .
  - A wakeboarder is attempting a trick while holding on to lines pointed upward .
  - A wakeboarder is jumping a huge wave .
  - person on a wakeboard in the air
```

Five people, one scene, five different words for it. The semantic view — which
does not make this mistake — arrives in Phase 2, once CLIP is available to
provide it.

---

## Reproducibility

Everything that can drift is pinned:

- **Dataset** — `jxie/flickr8k` at revision `56f58c96…`, verified after download
  by file count, per-split row count, and a content digest over all 8,000 JPEGs.
- **Models** — pinned to commit SHAs, not `main`. A Hugging Face model repo is
  mutable; `main` is not a version.
- **Split** — the standard Karpathy 6000/1000/1000, frozen to `data/splits.json`
  and committed, with a digest that detects hand edits.
- **Dependencies** — exact `==` pins in `requirements.txt`.

---

## Setup

```bash
conda create -n multimodal-search python=3.11 -y
conda activate multimodal-search

make install       # torch (CPU index) first, then everything else
make download      # ~1.1 GB of parquet, expands to 8,000 JPEGs
make test
```

Disk: roughly 6 GB total once model weights are cached.

**If the download stalls.** `huggingface_hub` restarts a file from zero when its
connection drops, so on a link that stalls mid-transfer a large file never
finishes. `make fetch` drives the transfer directly with HTTP range requests and
resumes from wherever it got to; run it first, then `make download`, which finds
the files already cached.

---

## Project layout

```
src/
  config.py            paths, seed, pinned revisions
  data/
    download.py        fetch, extract, verify, manifest
    dataset.py         ImageRecord: image + its 5 captions
    splits.py          frozen train/validation/test assignment
    agreement.py       inter-annotator agreement (the human ceiling)
  embedding/           CLIP encoder, cache, ChromaDB index   [Phase 2]
  search/              text→image, image→image               [Phase 3]
  vqa/                 BLIP wrapper, question typing         [Phase 5]
  eval/                metrics, latency, report generator    [Phase 4, 6]
  api.py               FastAPI endpoints                     [Phase 7]
eval_sets/             committed query and question sets
results/               committed metrics and plots
tests/
```

---

## Ownership

Personal project. Flickr8k is a public academic dataset. No employer data is
used anywhere in this repository.
