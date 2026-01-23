#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Excel-friendly merge (UTF-8 with BOM).

Merge ko translations from an "old" master_dialog.csv into a "new" master_dialog.csv
only when the Japanese line is identical (plus scenario_dir/Label/CharacterName).

Default behavior:
- Fill ONLY where new.ko is empty.
- Copy from old.ko when old.ko is non-empty.
- Match key: (scenario_dir, Label, CharacterName, ja)

Outputs:
- merged CSV encoded as UTF-8 with BOM (utf-8-sig) so Excel opens it cleanly.

Usage:
  python merge_ko_from_old.py --new master_dialog.csv --old master_dialog_old.csv --out master_dialog_merged.csv

Options:
  --overwrite   Overwrite new.ko even if it's not empty (default: off)
"""
import argparse
import csv
from collections import defaultdict

KEY_COLS = ("scenario_dir", "Label", "CharacterName", "ja")
KO_COL = "ko"

def norm(s: str) -> str:
    return "" if s is None else s

def build_old_map(old_rows):
    best_ko_by_key = {}
    dup_info = defaultdict(set)
    for r in old_rows:
        key = tuple(norm(r.get(c, "")) for c in KEY_COLS)
        ko = norm(r.get(KO_COL, ""))
        if ko:
            dup_info[key].add(ko)
            if key not in best_ko_by_key:
                best_ko_by_key[key] = ko
    return best_ko_by_key, dup_info

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, help="new CSV (post find/replace)")
    ap.add_argument("--old", required=True, help="old CSV (has existing ko translations)")
    ap.add_argument("--out", required=True, help="output merged CSV path")
    ap.add_argument("--overwrite", action="store_true", help="overwrite new.ko even if not empty")
    args = ap.parse_args()

    # Read old/new with UTF-8 BOM tolerance
    with open(args.old, "r", encoding="utf-8-sig", newline="") as f:
        old_reader = csv.DictReader(f)
        old_fieldnames = old_reader.fieldnames or []
        old_rows = list(old_reader)

    with open(args.new, "r", encoding="utf-8-sig", newline="") as f:
        new_reader = csv.DictReader(f)
        new_fieldnames = new_reader.fieldnames or []
        new_rows = list(new_reader)

    missing = [c for c in (*KEY_COLS, KO_COL) if c not in new_fieldnames or c not in old_fieldnames]
    if missing:
        raise SystemExit(
            f"[ERR] Missing required columns in CSV(s): {missing}\n"
            f"new has: {new_fieldnames}\nold has: {old_fieldnames}"
        )

    old_map, dup_info = build_old_map(old_rows)

    total = len(new_rows)
    keys_found = 0
    ko_filled = 0
    ko_skipped_existing = 0

    for r in new_rows:
        key = tuple(norm(r.get(c, "")) for c in KEY_COLS)
        old_ko = old_map.get(key)
        if not old_ko:
            continue

        keys_found += 1
        new_ko = norm(r.get(KO_COL, ""))
        if new_ko and not args.overwrite:
            ko_skipped_existing += 1
            continue

        r[KO_COL] = old_ko
        ko_filled += 1

    # Write output as UTF-8 with BOM so Excel doesn't 깨짐
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=new_fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(new_rows)

    dup_keys = [k for k, kos in dup_info.items() if len(kos) > 1]

    print("[OK] Merge done.")
    print(f"  total rows (new): {total}")
    print(f"  keys matched with old: {keys_found}")
    print(f"  ko filled from old: {ko_filled}")
    if not args.overwrite:
        print(f"  skipped (new.ko already had text): {ko_skipped_existing}")
    print(f"  note: old duplicate keys with multiple different ko: {len(dup_keys)}")

if __name__ == "__main__":
    main()
