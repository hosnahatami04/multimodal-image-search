"""Build and score the hard-negative set.

Retrieval over 8,000 images where only a handful are even vaguely relevant is
not a demanding test. The model can pick the right dog photograph out of a
corpus that is mostly beaches and street scenes by recognising "dog" alone. The
interesting question is what happens when every candidate is a dog photograph:
can the query's *specific* details still pick out the right one?

So groups of near-identical images are found by clustering the CLIP embeddings,
and each group is scored as a closed world of 5-10 candidates. The gap between
Recall@1 over the whole corpus and Recall@1 inside a group is the number this
phase exists to produce, and it is the one most projects never report.

The groups are found by clustering; the queries are written by hand. The first
attempt generated queries automatically from the words that separated a target
from its siblings, and produced ungrammatical strings like "a dog that is
hanging walks" -- which CLIP cannot parse, so every group scored zero and the
number measured the query generator rather than the model. Clustering to find
the groups and then writing the query by eye is what the plan asks for, and the
failed shortcut is why.

Run it with::

    python -m src.eval.hard_negatives --build --save
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass

import numpy as np

from src import config
from src.data.agreement import content_words
from src.data.dataset import load_records
from src.embedding import cache

logger = logging.getLogger(__name__)

# A group is only "hard" if its members are genuinely hard to tell apart.
# Random corpus pairs sit at 0.52; 0.85 is well into near-duplicate territory.
TIGHTNESS = 0.85

MIN_GROUP = 5
MAX_GROUP = 10

# Queries written by reading each group's captions and describing only the
# target, using details its siblings do not share. Keyed by target image id so
# a rebuild that reshuffles group numbering still matches them up.
HAND_WRITTEN_QUERIES: dict[str, str] = {
    # g01 -- snowboard/skateboard tricks in the air
    "test_00276": "a boy wearing a helmet airborne on a snowboard",
    # g02 -- bikes on dirt and rock
    "test_00082": "a dirt bike being ridden over bare rock",
    # g03 -- dogs in water
    "test_00230": "a dog swimming toward a waterfall",
    # g04 -- brown dogs on grass
    "test_00435": "a brown dog panting on grass in bright sunshine",
    # g05 -- dogs jumping obstacles
    "test_00150": "a brown dog clearing a jump on an agility course",
    # g06 -- skate ramps
    "test_00222": "a skateboarder dropping down a very steep outdoor ramp",
    # g07 -- jumping into water from a dock
    "test_00835": "a boy in a blue life vest leaping backwards off a pier",
    # g08 -- playground equipment
    "test_00724": "a child riding a blue and yellow merry-go-round",
    # g09 -- children in paddling pools
    "test_00146": "a boy in a pool with a beach ball floating behind him",
    # g10 -- acrobatics on the beach
    "test_00388": "a boy performing a backflip on sand",
    # g11 -- dogs running in fields
    "test_00243": "a white dog chasing a stuffed toy pulled along a string",
    # g12 -- snowy mountain camps
    "test_00448": "a man holding a flag beside tents pitched in snow",
}


@dataclass(frozen=True)
class HardGroup:
    """A cluster of near-identical images with a query targeting one of them."""

    group_id: str
    image_ids: tuple[str, ...]
    target_image_id: str
    query: str
    mean_similarity: float
    theme: str
    distinguishing: tuple[str, ...] = ()


def _distinguishing_words(target_captions, sibling_captions) -> list[str]:
    """Words the target's captions use that its near-duplicates do not.

    These are what a query has to lean on to separate the target from the rest
    of its group -- if there are none, the group has no answerable query and is
    discarded.
    """
    target_words: Counter[str] = Counter()
    for caption in target_captions:
        target_words.update(content_words(caption))

    sibling_words: set[str] = set()
    for captions in sibling_captions:
        for caption in captions:
            sibling_words |= content_words(caption)

    # Appearing in at least two of the five captions means it is a property of
    # the image, not one annotator's word choice.
    return [
        word
        for word, count in target_words.most_common()
        if count >= 2 and word not in sibling_words and len(word) > 2
    ]


def build_groups(
    tightness: float = TIGHTNESS, max_groups: int = 12, seed: int = config.SEED
) -> list[HardGroup]:
    """Cluster the embeddings and turn tight clusters into hard-negative groups."""
    entry = cache.load()
    records = {record.image_id: record for record in load_records()}

    vectors = entry.vectors
    ids = list(entry.image_ids)

    rng = np.random.default_rng(seed)
    # Seeding from the test split keeps the targets inside the held-out data.
    test_rows = [
        position
        for position, image_id in enumerate(ids)
        if records[image_id].split == config.EVAL_SPLIT
    ]
    rng.shuffle(test_rows)

    groups: list[HardGroup] = []
    claimed: set[str] = set()

    for seed_row in test_rows:
        if len(groups) >= max_groups:
            break

        target_id = ids[seed_row]
        if target_id in claimed:
            continue

        similarities = vectors @ vectors[seed_row]
        similarities[seed_row] = -np.inf

        neighbours = np.argsort(-similarities)[: MAX_GROUP - 1]
        neighbours = [row for row in neighbours if similarities[row] >= tightness]

        if len(neighbours) < MIN_GROUP - 1:
            continue
        if any(ids[row] in claimed for row in neighbours):
            continue

        member_ids = (target_id, *(ids[row] for row in neighbours))

        distinguishing = _distinguishing_words(
            records[target_id].captions,
            [records[ids[row]].captions for row in neighbours],
        )
        if len(distinguishing) < 2:
            # Nothing separates the target from its siblings in language, so no
            # fair query can be written for it.
            continue

        # Theme: the word every member shares, i.e. what the group is "of".
        shared: Counter[str] = Counter()
        for image_id in member_ids:
            words: set[str] = set()
            for caption in records[image_id].captions:
                words |= content_words(caption)
            shared.update(words)
        theme = next(
            (word for word, count in shared.most_common() if count == len(member_ids)),
            "scene",
        )

        groups.append(
            HardGroup(
                group_id=f"g{len(groups) + 1:02d}",
                image_ids=member_ids,
                target_image_id=target_id,
                query=HAND_WRITTEN_QUERIES.get(
                    target_id, " ".join(["a", theme, "that is", *distinguishing[:3]])
                ),
                mean_similarity=float(np.mean([similarities[row] for row in neighbours])),
                theme=theme,
                distinguishing=tuple(distinguishing[:5]),
            )
        )
        claimed.update(member_ids)

    return groups


def evaluate_groups(groups, searcher) -> dict:
    """Score each group as a closed world and compare against the open corpus.

    Two numbers per group: where the target ranks among its 5-10 near-identical
    siblings, and where it ranks among all 8,000. The first is the hard setting;
    the second is the easy one that most projects report alone.
    """
    from src.eval.retrieval_metrics import rank_of

    rows = []
    for group in groups:
        scores, ids = searcher.score_all(group.query)

        open_rank, _ = rank_of(scores, ids, group.target_image_id)

        # Closed world: only this group's members are candidates.
        positions = [ids.index(image_id) for image_id in group.image_ids]
        subset_scores = scores[positions]
        order = np.argsort(-subset_scores)
        closed_rank = int(np.flatnonzero(order == 0)[0]) + 1

        rows.append(
            {
                "group_id": group.group_id,
                "theme": group.theme,
                "query": group.query,
                "target_image_id": group.target_image_id,
                "group_size": len(group.image_ids),
                "mean_similarity": round(group.mean_similarity, 4),
                "rank_within_group": closed_rank,
                "rank_in_full_corpus": open_rank,
            }
        )

    hard_hits = sum(1 for row in rows if row["rank_within_group"] == 1)
    easy_hits = sum(1 for row in rows if row["rank_in_full_corpus"] == 1)

    return {
        "num_groups": len(rows),
        "mean_group_size": round(float(np.mean([r["group_size"] for r in rows])), 2),
        "mean_tightness": round(float(np.mean([r["mean_similarity"] for r in rows])), 4),
        "recall_at_1_within_group": round(hard_hits / len(rows), 4) if rows else 0.0,
        "recall_at_1_full_corpus": round(easy_hits / len(rows), 4) if rows else 0.0,
        "groups": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and score hard-negative groups.")
    parser.add_argument("--build", action="store_true", help="rebuild the groups from embeddings")
    parser.add_argument("--tightness", type=float, default=TIGHTNESS)
    parser.add_argument("--save", action="store_true", help="write the eval set and results")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    groups_path = config.EVAL_SETS_DIR / "hard_negatives.json"

    if args.build or not groups_path.exists():
        logger.info("clustering embeddings at tightness %.2f", args.tightness)
        groups = build_groups(tightness=args.tightness)
        if not groups:
            logger.error("no groups found; lower --tightness")
            return 1

        config.EVAL_SETS_DIR.mkdir(parents=True, exist_ok=True)
        groups_path.write_text(
            json.dumps(
                [
                    {
                        "group_id": group.group_id,
                        "theme": group.theme,
                        "query": group.query,
                        "target_image_id": group.target_image_id,
                        "image_ids": list(group.image_ids),
                        "mean_similarity": round(group.mean_similarity, 4),
                    }
                    for group in groups
                ],
                indent=1,
            ),
            encoding="utf-8",
        )
        logger.info("wrote %s (%d groups)", groups_path.name, len(groups))
    else:
        payload = json.loads(groups_path.read_text(encoding="utf-8"))
        groups = [
            HardGroup(
                group_id=item["group_id"],
                image_ids=tuple(item["image_ids"]),
                target_image_id=item["target_image_id"],
                query=item["query"],
                mean_similarity=item["mean_similarity"],
                theme=item["theme"],
            )
            for item in payload
        ]

    from src.search.text_to_image import TextToImageSearcher

    searcher = TextToImageSearcher(exact=True)
    report = evaluate_groups(groups, searcher)

    print(
        f"\n{report['num_groups']} groups, mean size {report['mean_group_size']}, "
        f"mean within-group similarity {report['mean_tightness']}\n"
    )
    print(f"  Recall@1 inside the group (hard)  : {report['recall_at_1_within_group']:.3f}")
    print(f"  Recall@1 over all 8,000   (easy)  : {report['recall_at_1_full_corpus']:.3f}")
    print(
        f"  gap                               : "
        f"{report['recall_at_1_within_group'] - report['recall_at_1_full_corpus']:+.3f}\n"
    )

    for row in report["groups"]:
        marker = "hit " if row["rank_within_group"] == 1 else "MISS"
        print(
            f"  {marker} {row['group_id']}  [{row['theme'][:12]:<12}] "
            f"in-group {row['rank_within_group']}/{row['group_size']}   "
            f"corpus {row['rank_in_full_corpus']:<6}  {row['query'][:44]}"
        )

    if args.save:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = config.RESULTS_DIR / "hard_negatives.json"
        output.write_text(json.dumps(report, indent=1), encoding="utf-8")
        logger.info("wrote %s", output.relative_to(config.PROJECT_ROOT))

    return 0


if __name__ == "__main__":
    sys.exit(main())
