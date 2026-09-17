"""Answer a natural-language question about an image.

BLIP-VQA is architecturally the opposite of CLIP, and the difference decides
what each one can be used for.

CLIP runs two separate encoders that never meet until a single dot product at
the end, so all 8,000 image vectors can be computed once and reused for every
query. BLIP fuses the image and the question *inside* the model: the question's
tokens attend over the image's patches, so the question can direct where the
model looks. That is what lets it answer "how many people are there" rather
than merely score how well that sentence matches the photograph.

The cost is that nothing can be precomputed. Every (image, question) pair is a
full forward pass, roughly a second on CPU, which is why BLIP answers questions
and CLIP does the searching rather than the other way round.

Generation is greedy and deterministic by construction: `do_sample=False` with
one beam. A VQA answer is one to three words, so there is nothing for sampling
to improve, and a non-deterministic scorer would make every accuracy number
unreproducible.

Run a check with::

    python -m src.vqa.blip_vqa --image data/raw/images/test_00000.jpg --question "how many dogs are there?"
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path

from src import config

logger = logging.getLogger(__name__)

# BLIP-VQA answers are short by design. Capping generation keeps a confused
# model from rambling into a sentence, which would then fail answer matching
# for a reason unrelated to whether it saw the image correctly.
MAX_ANSWER_TOKENS = 12

# Questions longer than this are truncated by the tokeniser. None of ours come
# close; the limit is stated so a pasted-in question cannot be silently cut.
MAX_QUESTION_TOKENS = 64


class VqaError(RuntimeError):
    """Raised when a question cannot be answered."""


class BlipVqa:
    """Wraps BLIP-VQA so callers only pass paths and strings.

    The model is loaded lazily and then kept: it is ~1.5 GB and takes several
    seconds to initialise, so loading it per request would make the API
    unusable, and loading it at import time would make the test suite
    uncollectable on a machine without the weights.
    """

    def __init__(
        self,
        model_id: str = config.BLIP_VQA_MODEL_ID,
        revision: str = config.BLIP_REVISION,
        batch_size: int = 8,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.batch_size = batch_size

    @cached_property
    def _loaded(self) -> tuple[object, object]:
        import torch
        from transformers import BlipForQuestionAnswering, BlipProcessor

        logger.info("loading %s @ %s", self.model_id, self.revision[:12])
        model = BlipForQuestionAnswering.from_pretrained(self.model_id, revision=self.revision)
        processor = BlipProcessor.from_pretrained(self.model_id, revision=self.revision)

        model.eval()
        torch.set_grad_enabled(False)
        return model, processor

    @property
    def model(self):
        return self._loaded[0]

    @property
    def processor(self):
        return self._loaded[1]

    # -----------------------------------------------------------------
    # Answering
    # -----------------------------------------------------------------

    def answer(self, image_path: Path | str, question: str) -> str:
        """Answer one question about one image.

        Raises:
            VqaError: if the image is missing or unreadable, or the question is
                empty.
        """
        return self.answer_batch([(image_path, question)])[0]

    def answer_batch(
        self,
        pairs: Sequence[tuple[Path | str, str]],
        *,
        show_progress: bool = False,
    ) -> list[str]:
        """Answer several (image, question) pairs.

        Batching matters here: each forward pass carries a fixed overhead, and
        100 questions answered one at a time spend most of their time on it.

        Returns:
            Answers in the order the pairs were given.
        """
        import torch
        from PIL import Image, UnidentifiedImageError

        if not pairs:
            return []

        blank = [position for position, (_, q) in enumerate(pairs) if not q.strip()]
        if blank:
            raise VqaError(f"empty question at positions {blank[:5]}")

        batches = range(0, len(pairs), self.batch_size)
        if show_progress:
            from tqdm import tqdm

            batches = tqdm(
                batches,
                total=(len(pairs) + self.batch_size - 1) // self.batch_size,
                desc="  answering",
                unit="batch",
            )

        answers: list[str] = []

        for start in batches:
            window = pairs[start : start + self.batch_size]

            images = []
            questions = []
            for image_path, question in window:
                path = Path(image_path)
                try:
                    with Image.open(path) as handle:
                        images.append(handle.convert("RGB"))
                except FileNotFoundError as error:
                    raise VqaError(f"image not found: {path}") from error
                except (UnidentifiedImageError, OSError) as error:
                    raise VqaError(f"cannot read image {path}: {error}") from error
                questions.append(question.strip())

            inputs = self.processor(
                images=images,
                text=questions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_QUESTION_TOKENS,
            )

            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_ANSWER_TOKENS,
                    # Greedy and single-beam: the answer is one to three words,
                    # so there is nothing for sampling or search to improve, and
                    # determinism is required for the accuracy numbers to be
                    # reproducible at all.
                    do_sample=False,
                    num_beams=1,
                )

            decoded = self.processor.batch_decode(generated, skip_special_tokens=True)
            answers.extend(text.strip() for text in decoded)

        return answers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask BLIP a question about an image.")
    parser.add_argument("--image", required=True, help="path to an image, or a corpus image id")
    parser.add_argument("--question", required=True, help="the question to ask")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    path = Path(args.image)
    if not path.is_file():
        path = config.IMAGES_DIR / f"{args.image}.jpg"

    try:
        answer = BlipVqa().answer(path, args.question)
    except VqaError as error:
        logger.error("%s", error)
        return 1

    print(f"\n  image:    {path.name}")
    print(f"  question: {args.question}")
    print(f"  answer:   {answer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
