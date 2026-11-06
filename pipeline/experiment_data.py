"""Convert a frozen committed dataset snapshot to bounded-RAM NumPy memmaps."""

import gzip
import hashlib
import json
from pathlib import Path
import random

import numpy as np


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def prepare_cache(data_dir, cache_dir, limits, seed):
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    raw = (data_dir / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema_version") not in (2, 3) or manifest["payoff"] != [[[6, 3], [3, 0]], [[8, 5], [5, 2]]]:
        raise ValueError("Unexpected schema/payoff; inspect the dataset before training.")
    window = manifest["config"]["window"]
    dimension = 4 * window + 10
    if manifest["mlp_schema"]["input_dimension"] != dimension:
        raise ValueError("Input schema does not match encoder.")
    fingerprint = hashlib.sha256(raw).hexdigest()
    identity = {"manifest_sha256": fingerprint, "limits": limits, "seed": seed}
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "cache.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["identity"] != identity:
            raise ValueError("Cache belongs to another snapshot/settings. Use a new --output directory, or use train/evaluate on the frozen cache.")
        print("Using existing packed cache", flush=True)
        return meta
    write_json(cache_dir / "source_manifest.json", manifest)
    meta = {"identity": identity, "window": window, "dimension": dimension,
            "source_config": manifest["config"], "splits": {}}
    for split in ("train", "validation", "test"):
        available = manifest["example_counts"].get(split, 0)
        capacity = min(limits[split] or available, available)
        if capacity <= 0:
            raise ValueError(f"No labeled examples in {split}")
        path = cache_dir / f"{split}.npy"
        packed = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                          shape=(capacity, dimension + 7))
        shards = [s for s in manifest["shards"] if s["split"] == split]
        random.Random(seed + {"train": 0, "validation": 1, "test": 2}[split]).shuffle(shards)
        count = 0
        try:
            for shard_number, shard in enumerate(shards):
                with gzip.open(data_dir / shard["file"], "rt", encoding="utf-8") as stream:
                    for line in stream:
                        record = json.loads(line)
                        for example in record["mlp_examples"]:
                            x, y = example["x"], example["y"]
                            se = example["defect_minus_cooperate_standard_error"]
                            if len(x) != dimension or len(y) != 6:
                                raise ValueError(f"Bad dimensions in {shard['file']}")
                            row = np.asarray(x + y + [se], dtype=np.float32)
                            if not np.isfinite(row).all() or se < 0:
                                raise ValueError("Nonfinite data or negative uncertainty")
                            if any(v not in (0, 1) for v in y[2:]) or sum(y[2:]) != 1:
                                raise ValueError("Opponent targets must be one-hot")
                            packed[count] = row
                            count += 1
                            if count == capacity:
                                break
                        if count == capacity:
                            break
                if shard_number % 100 == 0 or count == capacity:
                    print(f"Packing {split}: {count:,}/{capacity:,} examples", flush=True)
                if count == capacity:
                    break
            if count != capacity:
                raise ValueError(f"Manifest says {capacity} available {split} examples, found only {count}")
            packed.flush()
        finally:
            del packed
        meta["splits"][split] = {"file": path.name, "rows": count}
    write_json(meta_path, meta)  # Written only after all three splits are ready.
    return meta


def load_cache(cache_dir):
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "cache.json").read_text(encoding="utf-8"))
    arrays = {split: np.load(cache_dir / info["file"], mmap_mode="r")
              for split, info in meta["splits"].items()}
    return meta, arrays


def batches(array, batch_size, seed=None, block_rows=65536):
    """Block shuffle avoids random disk seeks and an all-data RAM copy."""
    rng = np.random.default_rng(seed)
    blocks = list(range(0, len(array), block_rows))
    if seed is not None:
        rng.shuffle(blocks)
    for start in blocks:
        block = np.array(array[start:start + block_rows], copy=True)
        if seed is not None:
            rng.shuffle(block, axis=0)
        for offset in range(0, len(block), batch_size):
            yield block[offset:offset + batch_size]
