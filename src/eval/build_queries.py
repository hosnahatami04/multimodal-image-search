"""Write the 50-query retrieval evaluation set.

Each query names one gold image from the held-out test split. The queries are
*rewritten* rather than copied from the captions, and that rewriting is the
whole point: a query that is verbatim a caption sitting in the index tests
string matching dressed up as semantic search. Every query here deliberately
reaches for different words than the caption it came from -- "airborne on a
snowboard" for "A snowboarder flies in the air" -- so a hit means the model
matched meaning, not surface form.

Each query also carries a `category`, so Phase 6 can report accuracy per query
type instead of one averaged number that hides which kinds of language the
model cannot handle. The categories:

* **object** -- a thing, plainly named. The baseline; failures here are serious.
* **action** -- what is happening, not just what is present.
* **scene** -- setting and context rather than a subject.
* **attribute** -- colour or property bound to a specific object. Probes the
  binding problem the Phase 3 smoke test already exposed.
* **counting** -- an exact quantity. A known CLIP weakness, included so the
  weakness is measured rather than avoided.
* **spatial** -- a relation between two things.
* **compositional** -- several constraints at once, all of which must hold.

Run it with::

    python -m src.eval.build_queries
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter

from src import config
from src.data.dataset import DatasetError, load_records

logger = logging.getLogger(__name__)

# (gold_image_id, rewritten query, category)
#
# Every gold id below was checked against the test split, and every query was
# written by reading that image's five captions and then describing the scene
# in different words.
QUERIES: tuple[tuple[str, str, str], ...] = (
    # --- object: a thing, plainly named ------------------------------------
    ("test_00927", "a dog leaping into water", "object"),
    ("test_00958", "a small white dog on a leash beside a fence", "object"),
    ("test_00685", "a black dog with a collar running across a lawn", "object"),
    ("test_00089", "a child holding sparklers", "object"),
    ("test_00534", "a motorcycle balanced on its rear wheel", "object"),
    ("test_00789", "a rowing boat on open blue water", "object"),
    ("test_00687", "a tent pitched high on a mountainside", "object"),
    ("test_00414", "a religious signboard on a street", "object"),
    # --- action: what is happening ----------------------------------------
    ("test_00314", "someone airborne on a snowboard", "action"),
    ("test_00247", "a wet dog shaking water off its coat", "action"),
    ("test_00845", "a dog catching a flying disc in mid-air", "action"),
    ("test_00488", "a person scaling a rock face", "action"),
    ("test_00415", "a girl leaping over a metal railing", "action"),
    ("test_00721", "a footballer striking the ball with his head", "action"),
    ("test_00581", "a kitesurfer losing balance over the sea", "action"),
    ("test_00241", "a dog jumping to grab a rope toy", "action"),
    ("test_00277", "a child crawling through a concrete tunnel", "action"),
    # --- scene: setting and context ---------------------------------------
    ("test_00131", "calm water under an evening sky", "scene"),
    ("test_00857", "onlookers watching fireworks in the dark", "scene"),
    ("test_00878", "people gathered around a campfire", "scene"),
    ("test_00609", "a large crowd at an outdoor gathering", "scene"),
    ("test_00114", "a child soaked by jets of water at a play park", "scene"),
    ("test_00852", "girls posing beside a decorated christmas tree", "scene"),
    ("test_00196", "a man sleeping rough on a pavement", "scene"),
    # --- attribute: colour or property bound to an object -----------------
    ("test_00194", "a skateboarder wearing a crimson top", "attribute"),
    ("test_00445", "an elderly man in a bright red cap", "attribute"),
    ("test_00617", "a dog wearing a blue garment on grass", "attribute"),
    ("test_00501", "a snowboarder dressed in vivid green", "attribute"),
    ("test_00069", "a black and white dog with a red collar", "attribute"),
    ("test_00946", "a man in red trunks jumping onto a board", "attribute"),
    ("test_00163", "twin girls in orange tops and green trousers", "attribute"),
    ("test_00218", "a dog carrying an orange ball while swimming", "attribute"),
    # --- counting: an exact quantity --------------------------------------
    ("test_00078", "three youngsters playing in beach sand", "counting"),
    ("test_00465", "three men assembling a sledge on snow", "counting"),
    ("test_00044", "three men gazing into the distance", "counting"),
    ("test_00090", "two boys eating ice lollies", "counting"),
    ("test_00143", "two dogs tumbling over each other among leaves", "counting"),
    ("test_00151", "two small girls playing together", "counting"),
    ("test_00783", "four or more swimmers heading toward a bridge", "counting"),
    # --- spatial: a relation between things -------------------------------
    ("test_00116", "one dog running past another beside patio chairs", "spatial"),
    ("test_00688", "a person suspended from a pole with mountains behind", "spatial"),
    ("test_00967", "a boy standing up on a railing", "spatial"),
    ("test_00460", "a child asleep stretched across two chairs", "spatial"),
    ("test_00792", "one climber seated on rock above the sea while another ascends", "spatial"),
    ("test_00215", "men on the shore with a boat out on the water", "spatial"),
    # --- compositional: several constraints at once -----------------------
    ("test_00774", "a cyclist in a blue helmet with feet up on the handlebars", "compositional"),
    ("test_00382", "a large brown dog chasing a smaller white one on grass", "compositional"),
    ("test_00866", "a man and a golden dog together on sand", "compositional"),
    ("test_00838", "a racing dog wearing a numbered bib", "compositional"),
    ("test_00487", "someone filming a skier performing a trick", "compositional"),
    ("test_00202", "a man gesturing toward a silver car", "compositional"),
)


def build() -> list[dict[str, str]]:
    """Validate every gold id against the dataset and return the query set.

    Raises:
        DatasetError: if a gold id is absent from the test split, or if a query
            happens to reproduce one of its gold image's captions verbatim.
    """
    records = {record.image_id: record for record in load_records(split=config.EVAL_SPLIT)}

    missing = [image_id for image_id, _, _ in QUERIES if image_id not in records]
    if missing:
        raise DatasetError(
            f"{len(missing)} gold ids are not in the {config.EVAL_SPLIT!r} split: "
            f"{', '.join(missing[:5])}"
        )

    duplicates = [
        image_id for image_id, count in Counter(i for i, _, _ in QUERIES).items() if count > 1
    ]
    if duplicates:
        raise DatasetError(f"gold ids used more than once: {duplicates}")

    entries: list[dict[str, str]] = []
    verbatim: list[str] = []

    for position, (image_id, text, category) in enumerate(QUERIES, start=1):
        normalised = " ".join(text.lower().split())
        for caption in records[image_id].captions:
            if " ".join(caption.lower().split()).rstrip(" .") == normalised.rstrip(" ."):
                verbatim.append(f"{image_id}: {text!r}")

        entries.append(
            {
                "query_id": f"q{position:03d}",
                "query": text,
                "gold_image_id": image_id,
                "category": category,
            }
        )

    if verbatim:
        raise DatasetError(
            "these queries reproduce a gold caption verbatim, which tests string "
            "matching rather than semantic search:\n  " + "\n  ".join(verbatim)
        )

    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the retrieval query set.")
    parser.add_argument("--show", action="store_true", help="print each query with its captions")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    try:
        entries = build()
    except DatasetError as error:
        logger.error("%s", error)
        return 1

    config.EVAL_SETS_DIR.mkdir(parents=True, exist_ok=True)
    output = config.EVAL_SETS_DIR / "retrieval_queries.json"
    output.write_text(json.dumps(entries, indent=1), encoding="utf-8")

    counts = Counter(entry["category"] for entry in entries)
    logger.info("wrote %s (%d queries)", output.name, len(entries))
    for category, count in sorted(counts.items()):
        logger.info("  %-14s %d", category, count)

    if args.show:
        records = {r.image_id: r for r in load_records(split=config.EVAL_SPLIT)}
        for entry in entries:
            print(f"\n{entry['query_id']}  [{entry['category']}]  {entry['query']}")
            print(f"  gold: {entry['gold_image_id']}")
            for caption in records[entry["gold_image_id"]].captions[:2]:
                print(f"    - {caption}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
