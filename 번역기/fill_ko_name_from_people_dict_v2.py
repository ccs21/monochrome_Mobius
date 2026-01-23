#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fill ko_name in name_map.csv by matching jp_name against people_dict_from_excel.yaml.

This YAML is expected to be like:
  'オシュトル':
    name_ko: 오슈토르
    gender: 남성
So we map jp -> value['name_ko'].

Outputs CSV as UTF-8 with BOM (utf-8-sig) for Excel.

Usage:
  python fill_ko_name_from_people_dict_v2.py --name_map name_map.csv --people_dict people_dict_from_excel.yaml --out name_map_filled_ko.csv

Options:
  --overwrite  overwrite existing ko_name values
  --keep_at    do NOT strip leading '@' when matching (default strips)
"""
import argparse
import csv
from pathlib import Path
from typing import Any, Dict

def norm_name(s: str, strip_at: bool) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if strip_at and s.startswith("@"):
        s = s[1:].strip()
    return s

def extract_jp_to_ko(obj: Any, out: Dict[str, str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                if isinstance(v, str):
                    out[k] = v
                elif isinstance(v, dict):
                    if "name_ko" in v and isinstance(v["name_ko"], str):
                        out[k] = v["name_ko"]
                    elif "ko" in v and isinstance(v["ko"], str):
                        out[k] = v["ko"]
                    elif "kr" in v and isinstance(v["kr"], str):
                        out[k] = v["kr"]
                extract_jp_to_ko(v, out)
            else:
                extract_jp_to_ko(v, out)
    elif isinstance(obj, list):
        for it in obj:
            extract_jp_to_ko(it, out)

def load_people_dict(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise SystemExit("[ERR] PyYAML is required for this people_dict format. Install: pip install pyyaml\n" + str(e))

    data = yaml.safe_load(text)
    jp2ko: Dict[str, str] = {}
    extract_jp_to_ko(data, jp2ko)
    return jp2ko

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name_map", required=True)
    ap.add_argument("--people_dict", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep_at", action="store_true")
    args = ap.parse_args()

    strip_at = not args.keep_at
    jp2ko_raw = load_people_dict(Path(args.people_dict))
    jp2ko = {norm_name(jp, strip_at): ko for jp, ko in jp2ko_raw.items() if isinstance(jp, str) and isinstance(ko, str)}

    with open(args.name_map, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        fields = r.fieldnames or []

    if "jp_name" not in fields:
        raise SystemExit("[ERR] name_map must contain 'jp_name' column")

    if "ko_name" not in fields:
        fields.append("ko_name")

    matched = 0
    written = 0
    for row in rows:
        jp = norm_name(row.get("jp_name", ""), strip_at)
        ko = jp2ko.get(jp, "")
        if ko:
            matched += 1
            cur = (row.get("ko_name") or "").strip()
            if (not cur) or args.overwrite:
                row["ko_name"] = ko
                written += 1
        else:
            row.setdefault("ko_name", row.get("ko_name", ""))

    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print("[OK] Done.")
    print(f"  matched jp_name in people_dict: {matched}")
    print(f"  ko_name written/updated: {written}")
    print(f"  out: {Path(args.out).resolve()}")

if __name__ == "__main__":
    main()
