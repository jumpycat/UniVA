"""Generate decoder architecture batches, adapted from decoders/gen_archv12.py."""

import argparse
import copy
import json
import random
from pathlib import Path

OPERATIONS = {
    "Upsample": ("bilinear", "nearest", "bicubic", "pixelshuffle"),
    "conv": ("WeightedConvPool3",),
    "act": ("LeakyReLU", "GELU", "SiLU"),
    "norm": ("BatchNorm2d", "InstanceNorm2d", "GroupNorm"),
}


def set_subblock(architecture, start, rng, active=True):
    """Fill three slots with a shuffled convolution/activation/normalization block."""
    operations = [
        (kind, rng.choice(OPERATIONS[kind])) for kind in ("conv", "act", "norm")
    ]
    rng.shuffle(operations)
    for offset, (kind, operation) in enumerate(operations):
        architecture[start + offset].update(
            type=kind if active else None, op=operation if active else None
        )


def sample_architecture(rng):
    """Sample one, two, or three consecutive upsampling stages ending at stage 3."""
    architecture = [dict(index=index + 1, type=None, op=None) for index in range(21)]
    for block in range(rng.randrange(3), 3):
        start = block * 7
        architecture[start].update(
            type="Upsample", op=rng.choice(OPERATIONS["Upsample"])
        )
        active_subblocks = rng.choice((1, 2))
        for subblock in range(active_subblocks):
            set_subblock(architecture, start + 1 + 3 * subblock, rng)
    return architecture


def generate_variant(base, modifications, rng):
    """Apply local changes using the original operation weights of 8:8:3:1."""
    architecture = copy.deepcopy(base)
    used_indices, used_subblocks = set(), set()
    successful = 0
    for _ in range(modifications * 1000):
        if successful == modifications:
            return architecture
        operation = rng.choices(
            ("upsample", "operator", "order", "depth"), (8, 8, 3, 1)
        )[0]
        active_blocks = [
            start for start in (0, 7, 14) if architecture[start]["type"] == "Upsample"
        ]
        active_subblocks = [
            start + offset
            for start in active_blocks
            for offset in (1, 4)
            if architecture[start + offset]["type"] is not None
        ]
        if operation in ("upsample", "operator"):
            kinds = {"Upsample"} if operation == "upsample" else {"conv", "act", "norm"}
            candidates = [
                index
                for index, layer in enumerate(architecture)
                if layer["type"] in kinds and index not in used_indices
            ]
            if not candidates:
                continue
            index = rng.choice(candidates)
            layer = architecture[index]
            alternatives = [
                value for value in OPERATIONS[layer["type"]] if value != layer["op"]
            ]
            if not alternatives:
                continue
            layer["op"] = rng.choice(alternatives)
            used_indices.add(index)
        elif operation == "order":
            candidates = [
                start for start in active_subblocks if start not in used_subblocks
            ]
            if not candidates:
                continue
            start = rng.choice(candidates)
            previous = [
                (layer["type"], layer["op"])
                for layer in architecture[start : start + 3]
            ]
            shuffled = previous.copy()
            # A successful order mutation must actually change the architecture.
            while shuffled == previous:
                rng.shuffle(shuffled)
            for offset, (kind, value) in enumerate(shuffled):
                architecture[start + offset].update(type=kind, op=value)
            used_subblocks.add(start)
        else:
            candidates = [
                start
                for start in active_blocks
                if not any(index // 7 == start // 7 for index in used_subblocks)
            ]
            if not candidates:
                continue
            start = rng.choice(candidates) + 4
            active = architecture[start]["type"] is None
            set_subblock(architecture, start, rng, active=active)
            used_subblocks.add(start)
        successful += 1
    raise RuntimeError(f"Could not apply {modifications} architecture modifications.")


def generate_batch(index, num_classes, random_batches, mixed_batches, rng):
    """Progress from random architectures to increasingly similar local variants."""
    base = sample_architecture(rng)
    records = [("base_arch", base)]
    seen = {json.dumps(base, sort_keys=True)}
    if index < random_batches:
        modification_counts = []
    elif index < random_batches + mixed_batches:
        modification_counts = [3] * min(3, num_classes - 1)
    else:
        modification_counts = ([2, 3, 3])[: num_classes - 1]

    for variant_index in range(num_classes - 1):
        modified = variant_index < len(modification_counts)
        for _ in range(1000):
            architecture = (
                generate_variant(base, modification_counts[variant_index], rng)
                if modified
                else sample_architecture(rng)
            )
            key = json.dumps(architecture, sort_keys=True)
            if key not in seen:
                seen.add(key)
                break
        else:
            raise RuntimeError(
                "Could not generate distinct architectures for this batch."
            )
        name = f"{'modified' if modified else 'random'}_variant_{variant_index + 1}"
        records.append((name, architecture))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/architectures"))
    parser.add_argument("--num-batches", type=int, default=12000)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--random-batches", type=int, default=4000)
    parser.add_argument("--mixed-batches", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.num_batches < 1 or args.num_classes < 2:
        parser.error("Use at least one batch and two classes per batch.")
    if min(args.random_batches, args.mixed_batches) < 0:
        parser.error("Curriculum stage lengths cannot be negative.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.glob("*.jsonl")):
        parser.error(
            "Output directory already contains JSONL files; use a fresh directory."
        )

    rng = random.Random(args.seed)
    for index in range(args.num_batches):
        records = generate_batch(
            index, args.num_classes, args.random_batches, args.mixed_batches, rng
        )
        with (args.output_dir / f"{index}.jsonl").open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")
        if (index + 1) % 1000 == 0 or index + 1 == args.num_batches:
            print(f"Generated {index + 1}/{args.num_batches} architecture batches.")


if __name__ == "__main__":
    main()
