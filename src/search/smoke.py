"""Run a fixed set of queries and print what comes back, for eyeballing.

Metrics can be healthy while the system is quietly broken. A preprocessing bug
that resizes images wrongly, or a missing normalisation, produces rankings that
are worse but still ordered -- Recall@5 drops from 0.7 to 0.4 and looks like a
model limitation rather than a bug. Looking at ten results catches that in a
minute, and there is no automated substitute for it.

The queries are chosen to probe specific things rather than to look good:

* **Plain object queries** should simply work. If "a dog running on grass"
  returns unrelated images, something is wrong with preprocessing.
* **Counting queries** probe a known CLIP weakness -- it has a weak grasp of
  exact quantity, and "three dogs" will often return two or four.
* **Colour + object** tests whether attributes bind to the right object.
* **Negation** is the sharpest known failure: CLIP has no real mechanism for
  "not", so "a street with no cars" tends to return streets full of cars. This
  is deliberately included so Phase 6 has the failure already on record.
* **Spatial relations** probe whether "behind" and "in front of" mean anything
  to the model.

Run it with::

    python -m src.search.smoke
    python -m src.search.smoke --k 5 --open   # also open the top hit
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import subprocess
import sys
from dataclasses import dataclass

from src.search.base import SearchError
from src.search.text_to_image import TextToImageSearcher

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SmokeQuery:
    """One query together with what it is supposed to reveal."""

    text: str
    probes: str


SMOKE_QUERIES: tuple[SmokeQuery, ...] = (
    SmokeQuery("a dog running on grass", "baseline -- must obviously work"),
    SmokeQuery("two children playing on a beach", "object + count + scene"),
    SmokeQuery("a man riding a bicycle", "object + action"),
    SmokeQuery("a woman in a red dress", "colour bound to the right object"),
    SmokeQuery("three dogs", "counting -- a known CLIP weakness"),
    SmokeQuery("a person snowboarding down a mountain", "action + setting"),
    SmokeQuery("a street with no cars", "negation -- CLIP has no mechanism for 'not'"),
    SmokeQuery("a dog behind a fence", "spatial relation"),
    SmokeQuery("someone cooking in a kitchen", "indoor scene, absent from most of Flickr8k"),
    SmokeQuery("a crowd of people at night", "scene + lighting"),
)


def run(k: int = 5, *, exact: bool = True, open_top: bool = False) -> int:
    """Print the top ``k`` results for every smoke query."""
    searcher = TextToImageSearcher(exact=exact)

    for number, query in enumerate(SMOKE_QUERIES, start=1):
        print(f"\n{'-' * 78}")
        print(f'{number:>2}. "{query.text}"')
        print(f"    probes: {query.probes}")
        print()

        try:
            results = searcher.search(query.text, k=k)
        except SearchError as error:
            print(f"    FAILED: {error}")
            continue

        for result in results:
            caption = result.captions[0] if result.captions else "(no caption)"
            print(f"    {result.rank}. {result.score:+.4f}  {result.image_id:<18}  {caption[:56]}")

        if open_top and results:
            # Windows-only convenience; suppressed elsewhere rather than failing
            # the whole smoke run over a missing shell.
            with contextlib.suppress(OSError):
                subprocess.run(["cmd", "/c", "start", "", str(results[0].path)], check=False)

    print(f"\n{'-' * 78}")
    print(
        "\nWhat to look for:\n"
        "  - The first few queries should return obviously matching photographs.\n"
        "    If they look random, the problem is image preprocessing or a missing\n"
        "    normalisation, not the model.\n"
        "  - 'three dogs' will probably return the wrong count. Expected.\n"
        "  - 'a street with no cars' will probably return streets WITH cars.\n"
        "    That is the negation failure, and it is worth reporting rather than\n"
        "    hiding.\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Eyeball a fixed set of search results.")
    parser.add_argument("--k", type=int, default=5, help="results per query")
    parser.add_argument("--chroma", action="store_true", help="query Chroma instead of the matrix")
    parser.add_argument("--open", action="store_true", help="open the top hit for each query")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")

    return run(k=args.k, exact=not args.chroma, open_top=args.open)


if __name__ == "__main__":
    sys.exit(main())
