"""Find images that look like another image.

Same index, same ranking, different query encoder. The query image goes through
CLIP's *image* encoder and is compared against the stored image vectors.

**This is semantic similarity, not pixel similarity**, and the distinction
surprises people often enough to be worth stating plainly. Two photographs of
different dogs on different lawns, shot on different cameras, share almost no
pixels -- yet they land close together, because CLIP was trained to place
"a dog on grass" near both. Conversely, an image and its own colour-inverted
copy are nearly identical pixel-wise and land far apart.

So this finds *photographs of the same kind of thing*, which is usually what
someone means by "similar", and never "the same file re-encoded", which is what
a perceptual hash would find.

An image is always its own nearest neighbour at similarity 1.0, so the query is
excluded from its own results by default.

Run it with::

    python -m src.search.image_to_image data/raw/images/test_00000.jpg
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from src.data.dataset import DatasetError
from src.search.base import BaseSearcher, SearchError, SearchResult

logger = logging.getLogger(__name__)


class ImageToImageSearcher(BaseSearcher):
    """Rank images by visual-semantic similarity to a query image."""

    def search(
        self,
        image_path: Path | str,
        k: int = 10,
        *,
        exclude_self: bool = True,
        restrict_to: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        """Return the ``k`` images most similar to the one at ``image_path``.

        Args:
            image_path: Any readable image, in the corpus or not.
            exclude_self: Drop the query image from its own results. Only has
                an effect when the query is a corpus image.

        Raises:
            SearchError: if the file does not exist.
        """
        path = Path(image_path)
        if not path.is_file():
            raise SearchError(f"image not found: {path}")

        vector = self.encoder.encode_image(path)
        exclude = path.stem if exclude_self else None

        return self.rank(vector, k=k, exclude=exclude, restrict_to=restrict_to)

    def search_by_id(
        self, image_id: str, k: int = 10, *, exclude_self: bool = True
    ) -> list[SearchResult]:
        """Search using a corpus image identified by id rather than by path."""
        record = self.records.get(image_id)
        if record is None:
            raise SearchError(f"unknown image id: {image_id}")

        try:
            path = record.require_file()
        except DatasetError as error:
            raise SearchError(str(error)) from error

        return self.search(path, k=k, exclude_self=exclude_self)

    def score_all(self, image_path: Path | str, *, exclude_self: bool = True):
        """Score every image against the query, for the evaluation harness."""
        path = Path(image_path)
        if not path.is_file():
            raise SearchError(f"image not found: {path}")

        vector = self.encoder.encode_image(path)
        return self.full_ranking(vector, exclude=path.stem if exclude_self else None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search images by example image.")
    parser.add_argument("image", help="path to the query image, or a corpus image id")
    parser.add_argument("-k", type=int, default=10, help="how many results")
    parser.add_argument("--exact", action="store_true", help="bypass Chroma, score exactly")
    parser.add_argument(
        "--include-self", action="store_true", help="do not drop the query from its results"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")

    searcher = ImageToImageSearcher(exact=args.exact)
    candidate = Path(args.image)

    try:
        if candidate.is_file():
            results = searcher.search(candidate, k=args.k, exclude_self=not args.include_self)
            label = candidate.name
        else:
            results = searcher.search_by_id(
                args.image, k=args.k, exclude_self=not args.include_self
            )
            label = args.image
    except SearchError as error:
        logger.error("%s", error)
        return 1

    print(f"\nsimilar to {label}\n")
    for result in results:
        print(result.describe())
    return 0


if __name__ == "__main__":
    sys.exit(main())
