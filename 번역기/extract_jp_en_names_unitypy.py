#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract JP/EN name (or term) pairs from Monochrome Mobius lang.ja/lang.en assetbundles.

This script uses UnityPy to parse UnityFS AssetBundles and tries to locate string tables
inside MonoBehaviours / ScriptableObjects. It then matches JP/EN tables by identical
typetree path and length, and outputs a CSV of pairs.

Install:
  pip install UnityPy

Usage:
  python extract_jp_en_names_unitypy.py --ja lang.ja --en lang.en --out name_map.csv

Optional:
  --debug : writes debug JSON dumps (candidates) next to output file.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# UnityPy is required on the user's environment
import UnityPy  # type: ignore


_JA_RE = re.compile(
    r"[\u3040-\u309F\u30A0-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]"
)

def contains_japanese(s: str) -> bool:
    return bool(_JA_RE.search(s))

def is_mostly_ascii(s: str) -> bool:
    if not s:
        return False
    ok = sum(1 for ch in s if ord(ch) < 128)
    return ok / max(1, len(s)) > 0.9

def walk_collect(node: Any, path: str, arrays: Dict[str, List[str]], maps: Dict[str, Dict[str, str]]) -> None:
    """
    Collect:
      - arrays[path] = list[str]  (string arrays)
      - maps[path]   = dict[str,str] (list-of-dict key/value tables)
    """
    if node is None:
        return

    # scalar string
    if isinstance(node, str):
        return

    # list
    if isinstance(node, list):
        if node and all(isinstance(x, str) for x in node):
            arrays[path] = node[:]  # copy
            return

        # list of dicts with key/value
        if node and all(isinstance(x, dict) for x in node):
            # common patterns: {"key": "...", "value": "..."} or {"m_Key": "...", "m_Value": "..."}
            key_fields = [("key", "value"), ("m_Key", "m_Value"), ("Key", "Value")]
            for kf, vf in key_fields:
                if all((kf in x and vf in x and isinstance(x[kf], str) and isinstance(x[vf], str)) for x in node):
                    table = {x[kf]: x[vf] for x in node}  # last wins
                    if table:
                        maps[path] = table
                    return

        # generic list traversal
        for idx, item in enumerate(node):
            walk_collect(item, f"{path}[{idx}]", arrays, maps)
        return

    # dict
    if isinstance(node, dict):
        for k, v in node.items():
            # skip huge raw byte arrays
            if isinstance(v, (bytes, bytearray)):
                continue
            walk_collect(v, f"{path}.{k}" if path else str(k), arrays, maps)
        return

    # other types
    return

def load_candidates(bundle_path: Path) -> Dict[str, Any]:
    env = UnityPy.load(str(bundle_path))
    arrays: Dict[str, List[str]] = {}
    maps: Dict[str, Dict[str, str]] = {}

    for obj in env.objects:
        try:
            # typetree works for MonoBehaviour / ScriptableObject; safe for others too
            data = obj.read_typetree()
        except Exception:
            continue

        # Record under a stable object label
        obj_label = f"{obj.type.name}:{obj.path_id}"
        try:
            m_name = None
            if isinstance(data, dict) and "m_Name" in data and isinstance(data["m_Name"], str):
                m_name = data["m_Name"]
            if m_name:
                obj_label = f"{obj_label}:{m_name}"
        except Exception:
            pass

        a2: Dict[str, List[str]] = {}
        m2: Dict[str, Dict[str, str]] = {}
        walk_collect(data, "", a2, m2)

        # namespace paths by object label to avoid collisions
        for p, arr in a2.items():
            arrays[f"{obj_label}::{p}"] = arr
        for p, mp in m2.items():
            maps[f"{obj_label}::{p}"] = mp

    return {"arrays": arrays, "maps": maps}

def score_pair_list(jp_list: List[str], en_list: List[str]) -> float:
    if not jp_list or not en_list or len(jp_list) != len(en_list):
        return -1.0
    n = len(jp_list)
    if n < 3:
        return -1.0

    jp_ja = sum(1 for s in jp_list if contains_japanese(s))
    en_ascii = sum(1 for s in en_list if is_mostly_ascii(s))

    # prefer bigger, cleaner signals
    return (jp_ja / n) * 2.0 + (en_ascii / n) * 1.5 + min(2.0, n / 200.0)

def score_pair_map(jp_map: Dict[str, str], en_map: Dict[str, str]) -> float:
    # match by identical keys intersection
    keys = set(jp_map.keys()) & set(en_map.keys())
    if len(keys) < 3:
        return -1.0
    jp_vals = [jp_map[k] for k in keys]
    en_vals = [en_map[k] for k in keys]
    jp_ja = sum(1 for s in jp_vals if contains_japanese(s))
    en_ascii = sum(1 for s in en_vals if is_mostly_ascii(s))
    n = len(keys)
    return (jp_ja / n) * 2.0 + (en_ascii / n) * 1.5 + min(2.0, n / 200.0)

def best_match(jp_cand: Dict[str, Any], en_cand: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Returns (mode, jp_key, en_key)
      mode = "array" or "map"
    """
    best = ("", "", "")
    best_score = -1.0

    jp_arrays: Dict[str, List[str]] = jp_cand["arrays"]
    en_arrays: Dict[str, List[str]] = en_cand["arrays"]
    # match arrays by exact candidate key (typetree path) and same length
    for k, jp_list in jp_arrays.items():
        en_list = en_arrays.get(k)
        if en_list is None:
            continue
        sc = score_pair_list(jp_list, en_list)
        if sc > best_score:
            best_score = sc
            best = ("array", k, k)

    jp_maps: Dict[str, Dict[str, str]] = jp_cand["maps"]
    en_maps: Dict[str, Dict[str, str]] = en_cand["maps"]
    for k, jp_map in jp_maps.items():
        en_map = en_maps.get(k)
        if en_map is None:
            continue
        sc = score_pair_map(jp_map, en_map)
        if sc > best_score:
            best_score = sc
            best = ("map", k, k)

    return best

def write_csv_pairs_array(out_path: Path, jp_key: str, en_key: str, jp_list: List[str], en_list: List[str]) -> None:
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "jp_name", "en_name", "source_key"])
        for i, (j, e) in enumerate(zip(jp_list, en_list)):
            # keep only plausible "name-like" items by default
            if not j or not e:
                continue
            # If you want ONLY character names, uncomment this heuristic:
            # if len(j) > 40 or len(e) > 40: continue
            w.writerow([i, j, e, jp_key])

def write_csv_pairs_map(out_path: Path, key: str, jp_map: Dict[str, str], en_map: Dict[str, str]) -> None:
    keys = sorted(set(jp_map.keys()) & set(en_map.keys()))
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id_key", "jp_name", "en_name", "source_key"])
        for k in keys:
            j = jp_map.get(k, "")
            e = en_map.get(k, "")
            if not j or not e:
                continue
            w.writerow([k, j, e, key])

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ja", required=True, help="path to lang.ja (UnityFS assetbundle)")
    ap.add_argument("--en", required=True, help="path to lang.en (UnityFS assetbundle)")
    ap.add_argument("--out", required=True, help="output CSV (utf-8-sig for Excel)")
    ap.add_argument("--debug", action="store_true", help="write debug candidate json files")
    args = ap.parse_args()

    ja_path = Path(args.ja)
    en_path = Path(args.en)
    out_path = Path(args.out)

    jp_cand = load_candidates(ja_path)
    en_cand = load_candidates(en_path)

    mode, jp_key, en_key = best_match(jp_cand, en_cand)

    if args.debug:
        out_base = out_path.with_suffix("")
        (out_base.parent / (out_base.name + ".ja_candidates.json")).write_text(
            json.dumps(jp_cand, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_base.parent / (out_base.name + ".en_candidates.json")).write_text(
            json.dumps(en_cand, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if mode == "array":
        jp_list = jp_cand["arrays"][jp_key]
        en_list = en_cand["arrays"][en_key]
        write_csv_pairs_array(out_path, jp_key, en_key, jp_list, en_list)
        print("[OK] extracted array pairs:", len(jp_list))
        print("  source:", jp_key)
        print("  out:", out_path)
        return

    if mode == "map":
        jp_map = jp_cand["maps"][jp_key]
        en_map = en_cand["maps"][en_key]
        write_csv_pairs_map(out_path, jp_key, jp_map, en_map)
        print("[OK] extracted map pairs:", len(set(jp_map) & set(en_map)))
        print("  source:", jp_key)
        print("  out:", out_path)
        return

    # nothing found: give hints
    print("[FAIL] could not locate a JP/EN string table with matching layout.")
    print("Try: --debug and inspect *.candidates.json to find the right typetree path.")
    print("If you find candidate keys, I can hard-code them for perfect extraction.")

if __name__ == "__main__":
    main()
