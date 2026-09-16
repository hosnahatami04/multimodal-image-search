"""Find images from a sentence.

The headline feature, and the one that only works because both modalities live
in the same space: the query string goes through CLIP's text encoder, and the
resulting vector is compared against image vectors directly. Nothing here
touches the captions -- an image with no caption at all would still be
retrievable, because the match is against what the photograph looks like, not
against words attached to it.

Run it with::

    python -m src.search.text_to_image "a dog jumping over a fence"
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

import numpy as np

from src.search.base import BaseSearcher, SearchError, SearchResult

logger = logging.getLogger(__name__)


class TextToImageSearcher(BaseSearcher):
    """Rank images by how well they match a text query."""

    def search(
        self,
        query: str,
        k: int = 10,
        *,
        restrict_to: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        """Return the ``k`` images best matching ``query``.

        Raises:
            SearchError: if the query is empty. An empty string produces a
                perfectly valid vector that ranks essentially arbitrarily,
                which is worse than an error.
        """
        if not query or not query.strip():
            raise SearchError("query is empty")

        vector = self.encoder.encode_text(query.strip())
        return self.rank(vector, k=k, restrict_to=restrict_to)

    def search_many(self, queries: Sequence[str], k: int = 10) -> list[list[SearchResult]]:
        """Run several queries, encoding them in one batch.

        The evaluation harness runs 50 queries at once; batching the text
        encoding is most of the saving, since each forward pass has fixed
        overhead regardless of how few strings it carries.
        """
        cleaned = [query.strip() for query in queries]
        blank = [position for position, query in enumerate(cleaned) if not query]
        if blank:
            raise SearchError(f"empty queries at positions {blank[:5]}")

        vectors = self.encoder.encode_texts(cleaned)
        return [self.rank(vector, k=k) for vector in vectors]

    def score_all(self, query: str, *, exclude: str | None = None):
        """Score every image against ``query``, for Recall@k and MRR."""
        if not query or not query.strip():
            raise SearchError("query is empty")

        vector = self.encoder.encode_text(query.strip())
        return self.full_ranking(vector, exclude=exclude)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search images by text.")
    parser.add_argument("query", nargs="+", help="the text to search for")
    parser.add_argument("-k", type=int, default=10, help="how many results")
    parser.add_argument("--exact", action="store_true", help="bypass Chroma, score exactly")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")

    query = " ".join(args.query)
    searcher = TextToImageSearcher(exact=args.exact)

    try:
        results = searcher.search(query, k=args.k)
    except SearchError as error:
        logger.error("%s", error)
        return 1

    print(f'\n"{query}"\n')
    for result in results:
        print(result.describe())

    scores = np.array([result.score for result in results])
    print(f"\n  score range {scores.min():+.4f} .. {scores.max():+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
