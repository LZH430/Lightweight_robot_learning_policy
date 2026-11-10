"""Split a converted G1 CSV by episode; no teacher dependencies required."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


def split_csv(source, output, seed=42, val_fraction=.2, dataset_name="unitree_g1_pack_camera"):
    source, output = Path(source), Path(output)
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    with source.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or not {"videoid", "data_dir", "instruction"} <= set(fields):
        raise ValueError("Expected converted WMA CSV with videoid, data_dir and instruction")
    if any(not r.get("videoid") or not r.get("data_dir") for r in rows):
        raise ValueError("Empty episode ID or data_dir")
    # videoid is the episode ID in the converted single-dataset CSV.
    ids = sorted({r["videoid"] for r in rows})
    if len(ids) < 2:
        raise ValueError("Need at least two episodes for disjoint train/validation")
    random.Random(seed).shuffle(ids)
    n = max(1, min(len(ids)-1, round(len(ids)*val_fraction)))
    val_ids = set(ids[:n])
    groups = {"train":[r for r in rows if r["videoid"] not in val_ids],
              "val":[r for r in rows if r["videoid"] in val_ids]}
    name = dataset_name
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Unsafe dataset_name")
    targets = [output/f"{name}_{s}.csv" for s in groups]+[output/f"{name}_split.json"]
    if any(p.exists() for p in targets):
        raise FileExistsError("Split outputs exist; choose another output directory")
    output.mkdir(parents=True, exist_ok=True)
    for split, selected in groups.items():
        with (output/f"{name}_{split}.csv").open("w",encoding="utf-8",newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
    meta = dict(source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                seed=seed,val_fraction=val_fraction,dataset_name=name,
                train_episode_ids=sorted(set(ids)-val_ids),val_episode_ids=sorted(val_ids),
                row_counts={s:len(r) for s,r in groups.items()})
    targets[-1].write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"train_episodes":len(ids)-n,"val_episodes":n,"rows":meta["row_counts"]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",required=True)
    parser.add_argument("--output",default="data/splits")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--val-fraction",type=float,default=.2)
    parser.add_argument("--dataset-name", default="unitree_g1_pack_camera")
    args = parser.parse_args()
    split_csv(args.csv,args.output,args.seed,args.val_fraction,args.dataset_name)


if __name__ == "__main__":
    main()
