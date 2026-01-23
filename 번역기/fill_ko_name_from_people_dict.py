#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fill ko_name in name_map.csv by matching jp_name against people_dict_from_excel.yaml (jp->kr).

- If ko_name column doesn't exist, it will be created.
- By default, existing non-empty ko_name values are kept. Use --overwrite to overwrite them.
- Normalization: trims whitespace, and optionally strips leading '@' (enabled by default).

Outputs CSV as UTF-8 with BOM (utf-8-sig) so Excel opens it cleanly.

Usage:
  python fill_ko_name_from_people_dict.py --name_map name_map.csv --people_dict people_dict_from_excel.yaml --out name_map_filled.csv

Options:
  --overwrite        overwrite existing ko_name even if non-empty
  --keep_at          do NOT strip leading '@' during matching
"""
import argparse
import csv
from pathlib import Path
from typing import Any, Dict

def load_yaml_mapping(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8-sig")

    # Prefer PyYAML if available
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text)
        mapping: Dict[str, str] = {}

        def harvest(obj: Any):
            nonlocal mapping
            if isinstance(obj, dict):
                # direct str->str mapping
                if all(isinstance(k, str) for k in obj.keys()) and all(isinstance(v, str) for v in obj.values()):
                    for k, v in obj.items():
                        if k:
                            mapping[str(k)] = str(v)
                # common nested keys
                for kk in ("people", "names", "person", "characters", "character_names"):
                    if kk in obj:
                        harvest(obj[kk])
                # recurse
                for v in obj.values():
                    harvest(v)
            elif isinstance(obj, list):
                for it in obj:
                    harvest(it)

        harvest(data)
        return mapping
    except Exception:
        pass

    # Fallback: simple "key: value" parser (flat mappings)
    mapping: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip().strip('"').strip("'")
        v = v.strip()
        if " #" in v:
            v = v.split(" #", 1)[0].rstrip()
        v = v.strip().strip('"').strip("'")
        if k:
            mapping[k] = v
    return mapping

def norm_name(s: str, strip_at: bool) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if strip_at and s.startswith("@"):
        s = s[1:].strip()
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name_map", required=True)
    ap.add_argument("--people_dict", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep_at", action="store_true", help="do not strip leading '@' when matching")
    args = ap.parse_args()

    people = load_yaml_mapping(Path(args.people_dict))
    # Build normalized lookup to tolerate '@' differences
    strip_at = not args.keep_at
    people_norm = {norm_name(jp, strip_at): kr for jp, kr in people.items()}

    with open(args.name_map, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        fields = r.fieldnames or []

    if "jp_name" not in fields:
        raise SystemExit("[ERR] name_map must contain 'jp_name' column")

    if "ko_name" not in fields:
        fields.append("ko_name")

    changed = 0
    matched = 0

    for row in rows:
        jp_raw = row.get("jp_name", "")
        jp = norm_name(jp_raw, strip_at)
        kr = people_norm.get(jp, "")
        if kr:
            matched += 1
            cur = (row.get("ko_name") or "").strip()
            if (not cur) or args.overwrite:
                row["ko_name"] = kr
                changed += 1
        else:
            # keep existing ko_name if any
            row.setdefault("ko_name", row.get("ko_name", ""))

    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print("[OK] Done.")
    print(f"  matched jp_name in people_dict: {matched}")
    print(f"  ko_name written/updated: {changed}")
    print(f"  out: {Path(args.out).resolve()}")

if __name__ == "__main__":
    main()
