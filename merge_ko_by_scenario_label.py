#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fill master_dialog.csv's ko column using a translated CSV, matching by (scenario_dir, Label).

- Key: (scenario_dir, Label)
- Default: fill only when master.ko is empty.
- Writes output as UTF-8 with BOM (utf-8-sig) so Excel opens it without garbling.

Usage:
  python merge_ko_by_scenario_label.py --master master_dialog.csv --translated ko_translated.csv --out master_dialog_filled.csv

Options:
  --overwrite    overwrite master.ko even if not empty
  --ko-col NAME  override translation column name (default: auto-detect ko/kr)
  --strip        strip whitespace around keys (default: on)
"""
import argparse
import csv
from collections import defaultdict

REQ_MASTER = ("scenario_dir", "Label")
DEFAULT_KO_CANDIDATES = ("ko", "kr", "KO", "KR")

def key_of(row, strip=True):
    sd = row.get("scenario_dir", "")
    lb = row.get("Label", "")
    if strip:
        sd = (sd or "").strip()
        lb = (lb or "").strip()
    return (sd or "", lb or "")

def detect_ko_col(fieldnames):
    for c in DEFAULT_KO_CANDIDATES:
        if c in fieldnames:
            return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--translated", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--ko-col", dest="ko_col", default=None, help="translation ko column name (auto-detect if omitted)")
    ap.add_argument("--strip", dest="strip", action="store_true", default=True, help="strip whitespace around keys (default: on)")
    ap.add_argument("--no-strip", dest="strip", action="store_false", help="do not strip whitespace in keys")
    args = ap.parse_args()

    # Read translated
    with open(args.translated, "r", encoding="utf-8-sig", newline="") as f:
        tr = csv.DictReader(f)
        t_fields = tr.fieldnames or []
        t_rows = list(tr)

    # Read master
    with open(args.master, "r", encoding="utf-8-sig", newline="") as f:
        mr = csv.DictReader(f)
        m_fields = mr.fieldnames or []
        m_rows = list(mr)

    for c in REQ_MASTER:
        if c not in m_fields or c not in t_fields:
            raise SystemExit(f"[ERR] Missing required column '{c}'. master has {m_fields}, translated has {t_fields}")

    if "ko" not in m_fields:
        raise SystemExit(f"[ERR] master CSV must contain 'ko' column. master has {m_fields}")

    ko_col = args.ko_col or detect_ko_col(t_fields)
    if not ko_col:
        raise SystemExit(f"[ERR] Could not detect translation ko column. Provide --ko-col. translated columns: {t_fields}")

    # Build translation map: key -> ko (first non-empty); track conflicts
    ko_by_key = {}
    conflicts = defaultdict(set)
    empty_ko_rows = 0

    for r in t_rows:
        k = key_of(r, strip=args.strip)
        v = (r.get(ko_col, "") or "")
        v = v.replace("\r\n", "\n").replace("\r", "\n")  # normalize newlines
        if not v.strip():
            empty_ko_rows += 1
            continue
        conflicts[k].add(v)
        if k not in ko_by_key:
            ko_by_key[k] = v

    conflict_keys = [k for k, vs in conflicts.items() if len(vs) > 1]

    # Merge into master
    total = len(m_rows)
    keys_matched = 0
    rows_filled = 0
    rows_skipped_existing = 0

    master_key_set = set()
    for r in m_rows:
        k = key_of(r, strip=args.strip)
        master_key_set.add(k)
        v = ko_by_key.get(k)
        if v is None:
            continue
        keys_matched += 1
        if (r.get("ko", "") or "").strip() and not args.overwrite:
            rows_skipped_existing += 1
            continue
        r["ko"] = v
        rows_filled += 1

    # Keys in translated but not in master
    translated_key_set = set(ko_by_key.keys())
    missing_in_master = sorted(list(translated_key_set - master_key_set))

    # Write output with BOM for Excel
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=m_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(m_rows)

    print("[OK] Merge done (scenario_dir + Label).")
    print(f"  master rows: {total}")
    print(f"  translated rows: {len(t_rows)} (empty ko rows ignored: {empty_ko_rows})")
    print(f"  matched keys (rows in master with a translation): {keys_matched}")
    print(f"  ko filled into master: {rows_filled}")
    if not args.overwrite:
        print(f"  skipped (master.ko already had text): {rows_skipped_existing}")
    print(f"  translation keys not found in master: {len(missing_in_master)}")
    print(f"  translation key conflicts (same key, different ko): {len(conflict_keys)}")
    if conflict_keys:
        print("  sample conflicts (up to 5):")
        for k in conflict_keys[:5]:
            sd, lb = k
            print("   -", {"scenario_dir": sd, "Label": lb})
            for v in list(conflicts[k])[:3]:
                print("      ko:", v[:120].replace("\n", "\\n") + ("..." if len(v) > 120 else ""))

if __name__ == "__main__":
    main()
