"""Turn images and text into vectors in one shared space.

CLIP carries two separate networks -- a vision transformer for images, a text
transformer for strings -- trained together so that a photograph and a sentence
describing it land close to each other. That shared geometry is what makes
"find images matching this sentence" a nearest-neighbour lookup rather than a
model run per candidate.

Two properties of this module matter downstream and are enforced here rather
than assumed:

**Every vector is L2-normalised.** CLIP similarity is cosine similarity. Once
vectors have unit length, the denominator of the cosine formula is 1 and the
similarity is a plain dot product -- so the whole 8,000-image search becomes one
matrix multiply. Normalising at write time also means nothing downstream can
forget to do it, which is the kind of mistake that degrades every result
quietly rather than raising.

**Batch and single-item encoding agree.** Encoding one image alone and encoding
it inside a batch of 32 must give the same vector. It does, but only because
the model is in eval mode with gradients off; the test suite pins this down.

Run a self-check with::

    python -m src.embedding.clip_encoder --check
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path

import numpy as np

from src import config

logger = logging.getLogger(__name__)

# CLIP's text encoder has a fixed 77-token context. Longer strings are
# truncated by the tokeniser, silently. Flickr8k captions top out around 38
# words so this never bites here, but a query pasted from elsewhere could.
MAX_TEXT_TOKENS = 77


def _features(output) -> object:
    """Pull the embedding tensor out of whatever the model version returns.

    transformers 4.x returned a bare tensor from ``get_text_features`` and
    ``get_image_features``. transformers 5.x returns a
    ``BaseModelOutputWithPooling`` whose ``pooler_output`` holds the same
    vector -- the projection head is applied inside the method, so despite the
    name it is the final embedding in the shared space, not the pre-projection
    pooled state.

    Supporting both keeps this module working across the version boundary, and
    the alternative -- reading ``.pooler_output`` at each call site -- would
    break silently on 4.x rather than raising.
    """
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        return pooled
    if isinstance(output, tuple):
        return output[0]
    return output


class EncoderError(RuntimeError):
    """Raised when encoding cannot proceed or produces something invalid."""


def l2_normalise(vectors: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Scale each vector to unit length.

    A zero vector has no direction to preserve, so its norm is clamped to 1
    rather than producing NaN. That case does not arise from CLIP in practice;
    the guard is here so a caller passing hand-built vectors gets a usable
    result instead of silent NaNs spreading through the index.
    """
    norms = np.linalg.norm(vectors, axis=axis, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vectors / norms).astype(np.float32)


class ClipEncoder:
    """Wraps CLIP so the rest of the project only sees ndarrays.

    The model is loaded lazily on first use: importing this module, and the
    modules that import it, stays cheap enough for the test suite to collect
    without pulling 600 MB of weights into memory.
    """

    def __init__(
        self,
        model_id: str = config.CLIP_MODEL_ID,
        revision: str = config.CLIP_REVISION,
        batch_size: int = config.ENCODE_BATCH_SIZE,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.batch_size = batch_size

    @cached_property
    def _loaded(self) -> tuple[object, object]:
        """Load model and processor once, in eval mode."""
        import torch
        from transformers import CLIPModel, CLIPProcessor

        logger.info("loading %s @ %s", self.model_id, self.revision[:12])
        model = CLIPModel.from_pretrained(self.model_id, revision=self.revision)
        processor = CLIPProcessor.from_pretrained(self.model_id, revision=self.revision)

        # eval() disables dropout, which would otherwise make encoding
        # non-deterministic and break the batch-equals-single guarantee.
        model.eval()
        torch.set_grad_enabled(False)

        return model, processor

    @property
    def model(self):
        return self._loaded[0]

    @property
    def processor(self):
        return self._loaded[1]

    @property
    def dim(self) -> int:
        """Width of the shared embedding space (512 for ViT-B/32)."""
        return int(self.model.config.projection_dim)

    # -----------------------------------------------------------------
    # Images
    # -----------------------------------------------------------------

    def encode_images(
        self,
        paths: Sequence[Path | str],
        *,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode image files into unit-norm vectors.

        Args:
            paths: Image file paths, in the order the rows should come back.
            show_progress: Draw a progress bar. Worth it for the full 8,000-image
                run, noise for anything smaller.

        Returns:
            Array of shape ``(len(paths), dim)``, dtype float32, every row unit
            length.

        Raises:
            EncoderError: if a file is missing or cannot be opened as an image.
        """
        import torch
        from PIL import Image, UnidentifiedImageError

        if not paths:
            return np.zeros((0, self.dim), dtype=np.float32)

        batches = range(0, len(paths), self.batch_size)
        if show_progress:
            from tqdm import tqdm

            batches = tqdm(
                batches,
                total=(len(paths) + self.batch_size - 1) // self.batch_size,
                desc="  encoding",
                unit="batch",
            )

        chunks: list[np.ndarray] = []

        for start in batches:
            window = paths[start : start + self.batch_size]
            images = []

            for path in window:
                path = Path(path)
                try:
                    with Image.open(path) as handle:
                        # CLIP's preprocessor expects RGB; a grayscale or
                        # palette image would otherwise produce the wrong
                        # channel count.
                        images.append(handle.convert("RGB"))
                except FileNotFoundError as error:
                    raise EncoderError(f"image not found: {path}") from error
                except (UnidentifiedImageError, OSError) as error:
                    raise EncoderError(f"cannot read image {path}: {error}") from error

            inputs = self.processor(images=images, return_tensors="pt")
            with torch.no_grad():
                output = self.model.get_image_features(**inputs)
            chunks.append(_features(output).cpu().numpy())

        return l2_normalise(np.concatenate(chunks, axis=0))

    # -----------------------------------------------------------------
    # Text
    # -----------------------------------------------------------------

    def encode_texts(
        self,
        texts: Sequence[str],
        *,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode strings into unit-norm vectors in the same space as images.

        Returns:
            Array of shape ``(len(texts), dim)``, dtype float32, unit rows.

        Raises:
            EncoderError: if any string is empty -- an empty query is a caller
                bug, and CLIP would happily return a vector for it.
        """
        import torch

        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        blank = [index for index, text in enumerate(texts) if not text.strip()]
        if blank:
            raise EncoderError(f"empty text at positions {blank[:5]}")

        batches = range(0, len(texts), self.batch_size)
        if show_progress:
            from tqdm import tqdm

            batches = tqdm(
                batches,
                total=(len(texts) + self.batch_size - 1) // self.batch_size,
                desc="  encoding",
                unit="batch",
            )

        chunks: list[np.ndarray] = []

        for start in batches:
            window = list(texts[start : start + self.batch_size])
            inputs = self.processor(
                text=window,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_TEXT_TOKENS,
            )
            with torch.no_grad():
                output = self.model.get_text_features(**inputs)
            chunks.append(_features(output).cpu().numpy())

        return l2_normalise(np.concatenate(chunks, axis=0))

    # -----------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------

    def encode_image(self, path: Path | str) -> np.ndarray:
        """Encode one image, returning a 1-D vector."""
        return self.encode_images([path])[0]

    def encode_text(self, text: str) -> np.ndarray:
        """Encode one string, returning a 1-D vector."""
        return self.encode_texts([text])[0]


def similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Cosine similarity between unit-norm vectors, as a dot product.

    Both sides are expected to be normalised already -- everything this module
    produces is. Accepts 1-D or 2-D on either side and returns the matrix of
    pairwise scores.
    """
    left = np.atleast_2d(left)
    right = np.atleast_2d(right)

    if left.shape[1] != right.shape[1]:
        raise EncoderError(f"dimension mismatch: {left.shape[1]} vs {right.shape[1]}")

    return left @ right.T


def _self_check() -> int:
    """Encode one image against its own caption and against a random one.

    This is the sanity check the project plan calls for before building
    anything on top of the embeddings: an image should sit closer to a caption
    that describes it than to an unrelated one. If it does not, preprocessing
    or normalisation is wrong, and every number computed later would be wrong
    with it.
    """
    from src.data.dataset import load_records

    records = load_records(split="test")
    subject, other = records[0], records[len(records) // 2]

    encoder = ClipEncoder()
    logger.info("embedding dim: %d", encoder.dim)

    image = encoder.encode_image(subject.require_file())
    own = encoder.encode_texts(list(subject.captions))
    foreign = encoder.encode_texts(list(other.captions))

    own_scores = similarity(image, own)[0]
    foreign_scores = similarity(image, foreign)[0]

    print(f"\nimage: {subject.image_id}")
    print(
        f"  its own captions      mean {own_scores.mean():+.4f}  "
        f"(min {own_scores.min():+.4f}, max {own_scores.max():+.4f})"
    )
    for score, caption in zip(own_scores, subject.captions, strict=True):
        print(f"    {score:+.4f}  {caption[:70]}")

    print(f"\n  captions of {other.image_id}  mean {foreign_scores.mean():+.4f}")
    for score, caption in zip(foreign_scores, other.captions, strict=True):
        print(f"    {score:+.4f}  {caption[:70]}")

    gap = own_scores.mean() - foreign_scores.mean()
    print(f"\n  gap: {gap:+.4f}")

    norm = float(np.linalg.norm(image))
    print(f"  image vector norm: {norm:.6f}  (must be 1.0)")

    if abs(norm - 1.0) > 1e-4:
        logger.error("vectors are not unit length -- normalisation is broken")
        return 1
    if gap <= 0:
        logger.error("image is no closer to its own captions than to random ones")
        return 1

    logger.info("self-check passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CLIP encoder utilities.")
    parser.add_argument(
        "--check", action="store_true", help="run the image-vs-own-caption sanity check"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    if args.check:
        return _self_check()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
