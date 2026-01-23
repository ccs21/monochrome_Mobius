#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract JP/EN character names from Monochrome Mobius MES lang bundles.

This version is tailored to the *actual* structure in your sample:
- Each lang.* file is a UnityFS bundle with a MonoBehaviour that has:
    root['param'] = list of dicts, each dict contains at least:
        {'ID': 'Chara_00000', 'Name': 'オシュトル', ...}
- lang.ja and lang.en have the same list length and share the same 'ID's.
So we join by ID and output jp/en name pairs.

Outputs CSV as UTF-8 with BOM (Excel-friendly).

Requirements:
  pip install UnityPy

Examples:
1) Single pair (your sample files):
  python extract_char_names_from_lang_bundles.py --ja lang.ja --en lang.en --out name_map.csv

2) Full scan:
  python extract_char_names_from_lang_bundles.py --mes_root "...\adv\mes" --out name_map.csv
"""
import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import UnityPy  # pip install UnityPy


def _find_param_table(env) -> Optional[List[Dict[str, Any]]]:
    """
    Find the MonoBehaviour typetree that contains a list-of-dicts under 'param'.
    Return that list.
    """
    for obj in env.objects:
        if obj.type.name != "MonoBehaviour":
            continue
        try:
            tree = obj.read_typetree()
        except Exception:
            continue
        if not isinstance(tree, dict):
            continue
        param = tree.get("param")
        if isinstance(param, list) and param and isinstance(param[0], dict) and ("ID" in param[0]) and ("Name" in param[0]):
            return param
    return None


def _load_id_to_name(bundle_path: Path) -> Dict[str, str]:
    env = UnityPy.load(str(bundle_path))
    param = _find_param_table(env)
    if not param:
        raise ValueError(f"No 'param' table found in: {bundle_path}")
    out: Dict[str, str] = {}
    for row in param:
        _id = str(row.get("ID", "")).strip()
        name = str(row.get("Name", "")).strip()
        if _id:
            out[_id] = name
    if not out:
        raise ValueError(f"'param' table present but empty in: {bundle_path}")
    return out


def extract_one_pair(ja_path: Path, en_path: Path) -> List[Dict[str, str]]:
    ja_map = _load_id_to_name(ja_path)
    en_map = _load_id_to_name(en_path)
    common = sorted(set(ja_map.keys()) & set(en_map.keys()))
    rows = []
    for _id in common:
        rows.append({
            "id": _id,
            "jp_name": ja_map.get(_id, ""),
            "en_name": en_map.get(_id, ""),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mes_root", help="MES root folder to scan recursively (contains many subfolders with lang.ja/lang.en)")
    ap.add_argument("--ja", help="Single lang.ja file (for sample test)")
    ap.add_argument("--en", help="Single lang.en file (for sample test)")
    ap.add_argument("--out", default="name_map.csv", help="Output CSV path")
    ap.add_argument("--dedupe", action="store_true", help="Deduplicate by (jp_name,en_name); keep one id and aggregate folders")
    args = ap.parse_args()

    out_rows: List[Dict[str, str]] = []
    # For dedupe aggregation
    agg = defaultdict(lambda: {"id": "", "jp_name": "", "en_name": "", "folders": set()})

    if args.ja and args.en:
        rows = extract_one_pair(Path(args.ja), Path(args.en))
        for r in rows:
            r["folder"] = ""
            out_rows.append(r)

    if args.mes_root:
        mes_root = Path(args.mes_root)
        scanned = 0
        ok = 0
        for root, _, files in os.walk(mes_root):
            if "lang.ja" not in files or "lang.en" not in files:
                continue
            scanned += 1
            root_p = Path(root)
            ja_path = root_p / "lang.ja"
            en_path = root_p / "lang.en"
            try:
                rows = extract_one_pair(ja_path, en_path)
                ok += 1
                for r in rows:
                    r["folder"] = str(root_p)
                    out_rows.append(r)
            except Exception:
                # silently skip folders that aren't the expected table structure
                continue
        print(f"[INFO] scanned folder pairs: {scanned}, extracted from: {ok}")

    if not out_rows:
        raise SystemExit("[FAIL] No rows extracted. (The bundles may not contain the expected 'param' table.)")

    out_path = Path(args.out)

    if args.dedupe:
        for r in out_rows:
            key = (r["jp_name"], r["en_name"])
            d = agg[key]
            if not d["id"]:
                d["id"] = r["id"]
                d["jp_name"] = r["jp_name"]
                d["en_name"] = r["en_name"]
            if r.get("folder"):
                d["folders"].add(r["folder"])
        final_rows = []
        for (jp, en), d in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1])):
            final_rows.append({
                "id": d["id"],
                "jp_name": jp,
                "en_name": en,
                "folders": " | ".join(sorted(d["folders"]))[:2000],
            })
        fieldnames = ["id", "jp_name", "en_name", "folders"]
    else:
        final_rows = out_rows
        fieldnames = ["id", "jp_name", "en_name", "folder"]

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(final_rows)

    print("[OK] Wrote:", out_path.resolve())
    print("rows:", len(final_rows))


if __name__ == "__main__":
    main()
