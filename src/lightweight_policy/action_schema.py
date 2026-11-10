"""Validate dataset action order before loading the teacher (standard library only)."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_schema(path):
    path = Path(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    result = {"source": str(path.resolve()), "sha256": sha256(path)}
    for key in ("action", "observation.state"):
        feature = obj["features"][key]
        names = feature.get("names")
        shape = feature.get("shape")
        if (
            isinstance(names, list)
            and len(names) == 1
            and isinstance(names[0], list)
        ):
            names = names[0]

        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(n, str) and n.strip() for n in names)
        ):
            raise ValueError(f"{path}: {key} requires an ordered list of non-empty names")
        if shape != [len(names)] or len(set(names)) != len(names):
            raise ValueError(f"{path}: {key} shape/names mismatch or duplicate names")
        result[key] = {"names": names, "shape": shape}
    return result


def dataset_entries(config):
    if set(config["datasets"]) != {"train", "val"}:
        raise ValueError("Exactly train and val splits are required")
    for split, value in config["datasets"].items():
        entries = value if isinstance(value,list) else [value]
        if not entries:
            raise ValueError(f"{split}: empty dataset list")
        seen = set()
        for params in entries:
            name = params["dataset_name"]
            if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
                raise ValueError("dataset_name must be a safe filename prefix")
            if name in seen:
                raise ValueError(f"Duplicate dataset {name} in {split}")
            seen.add(name)
            yield split, params


def verify(config):
    sources = config.get("schema_sources", {})
    required = {p["dataset_name"] for _, p in dataset_entries(config)}
    if not required <= sources.keys():
        raise ValueError(f"Missing schema_sources for {sorted(required - sources.keys())}")
    schemas = {name: load_schema(path) for name,path in sources.items()}
    reference = next(iter(schemas.values()))
    for name,schema in schemas.items():
        for key in ("action", "observation.state"):
            if schema[key] != reference[key]:
                raise ValueError(f"Ordered {key} schema differs for {name}; do not mix these datasets")
    for split,params in dataset_entries(config):
        schema = schemas[params["dataset_name"]]
        for key,dim in (("action","max_action_dim"),("observation.state","max_state_dim")):
            if len(schema[key]["names"]) != params[dim]:
                raise ValueError(f"{split}: this exporter requires exact named dimensions; {dim} differs")
    return schemas


def normalization_sources(config):
    result = {}
    for split,params in dataset_entries(config):
        root = params.get("transition_dir")
        if not root:
            raise ValueError("Explicit local transition_dir required to record normalization provenance")
        path = Path(root)/params["dataset_name"]/"meta_data"/"stats.safetensors"
        result.setdefault(split, {})[params["dataset_name"]] = {"source":str(path.resolve()), "sha256":sha256(path),
                         "mode":params["normalization_mode"]}
    for name in result["train"].keys() & result["val"].keys():
        train, val = result["train"][name], result["val"][name]
        if (train["sha256"],train["mode"]) != (val["sha256"],val["mode"]):
            raise ValueError(f"Train/validation normalization differs for {name}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    schemas = verify(config)
    print(json.dumps(schemas,indent=2,ensure_ascii=False))
    print("PASS: named action/state order agrees; checkpoint semantics and units are not verified")


if __name__ == "__main__":
    main()
