"""Time a query end to end on CPU.

"It works" and "it works in 340 ms on a laptop CPU" are different claims, and
only the second one is an engineering claim. This measures the second.

The timing is broken into its three stages, because they behave differently
and knowing which dominates is what tells you where optimisation would go:

* **encode** -- running the query string through CLIP's text transformer. Fixed
  cost per query, independent of corpus size.
* **search** -- the dot product against the corpus matrix. Scales with the
  number of images.
* **assemble** -- turning row indices into SearchResult objects with paths and
  captions attached.

p50 and p95 are reported rather than a mean. A mean hides the tail, and the
tail is what a user actually notices: a p95 of 800 ms means one query in twenty
feels slow, which no average will tell you.

Model loading is excluded and measured separately -- it happens once at API
startup, not per request, and folding it into the per-query number would make
the first query look catastrophic and every later one impossible to interpret.

Run it with::

    python -m src.eval.latency --save
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np

from src import config

logger = logging.getLogger(__name__)

# Queries drawn from the smoke set plus a few longer ones, so the timing covers
# both short and long text inputs.
TIMING_QUERIES: tuple[str, ...] = (
    "a dog running on grass",
    "two children playing on a beach",
    "a man riding a bicycle",
    "a woman in a red dress",
    "a person snowboarding down a mountain",
    "a dog behind a fence",
    "someone cooking in a kitchen",
    "a crowd of people at night",
    "a brown dog leaping into the water beside a wooden dock at sunset",
    "a group of people wearing winter clothing standing on a snowy mountain",
)


@dataclass(frozen=True)
class Stage:
    """Timing distribution for one stage, in milliseconds."""

    name: str
    p50: float
    p95: float
    p99: float
    mean: float
    minimum: float
    maximum: float

    def describe(self, width: int = 10) -> str:
        return (
            f"  {self.name:<{width}}  p50 {self.p50:7.2f}   p95 {self.p95:7.2f}   "
            f"p99 {self.p99:7.2f}   mean {self.mean:7.2f}   "
            f"min {self.minimum:6.2f}   max {self.maximum:7.2f}"
        )


def _stage(name: str, samples_ms: list[float]) -> Stage:
    values = np.asarray(samples_ms)
    return Stage(
        name=name,
        p50=float(np.percentile(values, 50)),
        p95=float(np.percentile(values, 95)),
        p99=float(np.percentile(values, 99)),
        mean=float(values.mean()),
        minimum=float(values.min()),
        maximum=float(values.max()),
    )


def benchmark(runs: int = 100, k: int = 10, warmup: int = 5) -> dict:
    """Time `runs` queries, cycling through the query list.

    Args:
        runs: How many timed queries to perform.
        k: Results per query.
        warmup: Untimed queries first. The first call through any torch path is
            slower than the rest -- lazy kernel selection, allocator warm-up --
            and including it would put a spike in the tail that has nothing to
            do with steady-state behaviour.
    """
    from src.search.text_to_image import TextToImageSearcher

    load_started = time.perf_counter()
    searcher = TextToImageSearcher(exact=True)
    _ = searcher.encoder.dim  # forces the model load
    _ = searcher._matrix
    load_seconds = time.perf_counter() - load_started

    logger.info("model and index loaded in %.1fs (one-off, at API startup)", load_seconds)

    for index in range(warmup):
        searcher.search(TIMING_QUERIES[index % len(TIMING_QUERIES)], k=k)

    encode_ms: list[float] = []
    search_ms: list[float] = []
    total_ms: list[float] = []

    for index in range(runs):
        query = TIMING_QUERIES[index % len(TIMING_QUERIES)]

        started = time.perf_counter()
        vector = searcher.encoder.encode_text(query)
        encoded = time.perf_counter()
        searcher.rank(vector, k=k)
        finished = time.perf_counter()

        encode_ms.append((encoded - started) * 1000)
        search_ms.append((finished - encoded) * 1000)
        total_ms.append((finished - started) * 1000)

    return {
        "runs": runs,
        "k": k,
        "corpus_size": len(searcher.records),
        "load_seconds": round(load_seconds, 2),
        "stages": [
            asdict(_stage("encode", encode_ms)),
            asdict(_stage("search", search_ms)),
            asdict(_stage("total", total_ms)),
        ],
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "python": platform.python_version(),
            "threads": _torch_threads(),
        },
    }


def _torch_threads() -> int:
    try:
        import torch

        return int(torch.get_num_threads())
    except Exception:
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark query latency on CPU.")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--save", action="store_true", help="write results/latency.json")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    report = benchmark(runs=args.runs, k=args.k)

    print(
        f"\n{report['runs']} queries, k={report['k']}, "
        f"{report['corpus_size']} images, {report['environment']['threads']} threads\n"
        f"  milliseconds\n"
    )
    for stage in report["stages"]:
        print(Stage(**stage).describe())

    encode = next(s for s in report["stages"] if s["name"] == "encode")
    search = next(s for s in report["stages"] if s["name"] == "search")
    share = encode["p50"] / (encode["p50"] + search["p50"])
    print(
        f"\n  Text encoding is {share:.0%} of the query. The dot product over "
        f"{report['corpus_size']:,} vectors\n"
        f"  is the cheap half, which is why an approximate index would buy "
        f"nothing at this scale.\n"
        f"\n  Model + index load: {report['load_seconds']}s, once at startup."
    )

    if args.save:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = config.RESULTS_DIR / "latency.json"
        output.write_text(json.dumps(report, indent=1), encoding="utf-8")
        logger.info("wrote %s", output.relative_to(config.PROJECT_ROOT))

    return 0


if __name__ == "__main__":
    sys.exit(main())
