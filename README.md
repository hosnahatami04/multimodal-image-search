# multimodal-image-search

Text-to-image and image-to-image search over Flickr8k, plus visual question
answering. CPU only.

Search runs on CLIP (`ViT-B/32`), question answering on BLIP (`blip-vqa-base`).
Neither is fine-tuned — both run zero-shot on the Flickr8k test split. The
repository includes the evaluation code, the hand-written query and question
sets, and a breakdown of the failure cases by category.

Full numbers and plots: [`results/report.md`](results/report.md).

## Results

8,000 images, 40,000 captions, Karpathy test split.

| Retrieval | |
|---|---|
| Recall@5 | 0.431 |
| Recall@10 | 0.569 |
| MRR | 0.261 |
| Recall@1 | 0.118 (0.510 counting same-scene matches) |
| Latency | 7 ms per query |

| VQA | |
|---|---|
| Overall | 0.790 |
| Object presence | 0.941 |
| Counting | 0.882 |
| Action recognition | 0.412 |
| Latency | 370 ms per question |

Two notes on reading these:

Recall assumes one correct answer per query. The corpus has many near-duplicate
scenes, so a returned image that matches the query but isn't the labelled answer
counts as a miss. `src/eval/false_negatives.py` measures how often that happens;
Recall@1 goes from 0.118 to 0.510 when those are counted as correct.

Five annotators captioned each image, so inter-annotator agreement is measurable:
0.823 subject consensus. Roughly 5% of images are ones where the five captions
disagree about the subject.

## Installation

Python 3.11, ~4 GB free disk.

```bash
git clone https://github.com/hosnahatami04/multimodal-image-search.git
cd multimodal-image-search
make install
```

torch must come from the CPU index — a plain `pip install torch` pulls the 2.5 GB
CUDA build. `make install` handles the ordering; by hand it is:

```bash
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Download the dataset and build the index. Once, and slow:

```bash
make download   # ~1 GB
make encode     # ~3 min
```

`make fetch` is a resumable alternative if the Hub client stalls on the download.

Run the API:

```bash
make serve      # http://127.0.0.1:8000/docs
```

```bash
curl -X POST "http://127.0.0.1:8000/search/text?k=5" -F "query=a dog running on grass"
curl -X POST "http://127.0.0.1:8000/search/image?k=5" -F "image=@photo.jpg"
curl -X POST "http://127.0.0.1:8000/vqa" -F "image=@photo.jpg" -F "question=How many dogs are there?"
```

BLIP loads on first `/vqa` call, so that call takes ~1.5 s and subsequent ones
~370 ms. Uploads are capped at 12 MB.

### Docker

```bash
make docker-build
make docker-run   # http://localhost:8000/docs
```

Weights are baked into the image and `HF_HUB_OFFLINE=1` is set, so the container
needs no network. The build reads `data/embeddings` and `data/chroma`, so run
`make encode` first.

## Development

```bash
make test       # 226 tests, no weights or dataset required — what CI runs
make test-all   # 259 tests, needs CLIP and the dataset
make lint
make format
```

Most tests run against a synthetic 9-image fixture in `tests/conftest.py`.

`make help` lists all targets:

| | |
|---|---|
| `make evaluate` | full evaluation pipeline |
| `make eval` | retrieval metrics |
| `make vqa` | VQA metrics |
| `make failures` | failure analysis and worked cases |
| `make report` | regenerate `results/report.md` |
| `make check` | metric regression check |
| `make smoke` | manual check over fixed queries |

### Layout

```
src/
  data/        download, dataset, splits, annotator agreement
  embedding/   CLIP encoder, vector cache, index building
  search/      text→image, image→image
  eval/        retrieval metrics, VQA metrics, failure analysis, plots
  vqa/         BLIP wrapper, answer matching, question set
  api.py       FastAPI service
eval_sets/     51 queries, 100 questions
results/       committed metrics and report
```

### Metric gate

CI runs `scripts/check_metrics.py`, which fails if any headline metric drops
below its floor. It also checks that result files are consistent with each other
and were produced at the pinned model revisions.

A model regression raises no exception and breaks no test, so without this it
lands silently. If a change legitimately moves the numbers, update the floors in
the same commit.

## Contributing

Open an issue before starting anything substantial.

- Run `make lint` and `make test` before opening a PR.
- Branch off current `main`. Branching off an older commit produces CI failures
  that look like real bugs.
- If a change affects the metrics, run `make evaluate` and commit the updated
  result files rather than editing them.
- Query and question sets in `eval_sets/` are hand-written. Generated queries
  were tried and produced ungrammatical sentences that scored zero.
- Model revisions are pinned to commit SHAs. Hub repositories are mutable.

## Known issues

- **Counting above two is unreliable.** 14/14 correct for one and two, degrading
  to approximate above that, usually off by one.
- **Multi-constraint queries drop constraints.** "A cyclist in a blue helmet with
  his feet on the handlebars" returns cyclists; the helmet and feet are ignored.
  The sentence is compressed into a single vector.
- **No negation.** "A street with no cars" returns streets with cars.
- **Action recognition is weak** (0.412). A girl whistling is read as "talking on
  the phone" — small object near a face.
- **Colour under warm lighting.** Red is systematically read as brown.
- **Raw similarity scores are not comparable across queries.** Unrelated image
  pairs score 0.52, not 0; CLIP's embeddings occupy a narrow cone. Rankings are
  meaningful, absolute values are not — don't threshold on them.
- **One worker per container.** Each holds ~2 GB of model weights.
- **English only.**
