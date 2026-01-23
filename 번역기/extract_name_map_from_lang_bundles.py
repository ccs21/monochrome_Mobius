#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan MES subfolders for lang.ja + lang.en Unity AssetBundles, extract translation tables,
and build JP/EN name pairs. Then, apply a user people dictionary (jp->kr) to produce KR names.

Key idea:
- Parse TextAsset contents from each bundle (CSV/TSV/JSON/key=value supported).
- Align ja/en by shared keys.
- Treat entries as "character name" when:
    (A) ja_value exactly matches a jp name in people_dict, OR
    (B) key looks like a name key (contains 'name'/'chara' etc.) and value is short.

Outputs:
- CSV (UTF-8 with BOM) with jp/en/kr, plus where it came from.
- YAML (jp->kr) you can paste into your translator dictionary.

Requires:
- UnityPy (recommended). Install: pip install UnityPy
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

# ---- YAML loader (no external deps) ----
def load_simple_yaml_mapping(path: Path) -> dict[str, str]:
    """
    Minimal YAML mapping reader for files like:
      "JP": "KR"
      JP2: KR2
    Supports quotes and escapes modestly. Ignores nested structures.
    If you already use PyYAML, you can switch to yaml.safe_load.
    """
    text = path.read_text(encoding="utf-8-sig")
    mapping: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip().strip('"').strip("'")
        v = v.strip()
        # remove inline comments
        if " #" in v:
            v = v.split(" #", 1)[0].rstrip()
        v = v.strip().strip('"').strip("'")
        if k:
            mapping[k] = v
    return mapping

# ---- TextAsset parsers ----
def parse_kv_lines(text: str) -> dict[str, str] | None:
    # key=value per line
    if "=" not in text:
        return None
    out = {}
    ok = 0
    for line in text.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k:
            out[k] = v
            ok += 1
    return out if ok >= 10 else None  # avoid false positives

def parse_csv_tsv(text: str) -> dict[str, str] | None:
    # Heuristic: contains many tab/comma delimiters and at least 2 columns.
    delim = "\t" if text.count("\t") > text.count(",") else ","
    if (delim == "\t" and text.count("\t") < 20) or (delim == "," and text.count(",") < 20):
        return None
    lines = text.splitlines()
    if len(lines) < 2:
        return None
    # Try csv
    try:
        import io
        f = io.StringIO(text)
        reader = csv.reader(f, delimiter=delim)
        rows = list(reader)
        if not rows or len(rows[0]) < 2:
            return None
        # Common layouts: key,value or id,text etc.
        # Find best key/text columns by header.
        header = [c.strip().lower() for c in rows[0]]
        key_idx = None
        val_idx = None
        for i, h in enumerate(header):
            if h in ("key", "id", "label", "name", "string_id", "stringid"):
                key_idx = i
            if h in ("text", "value", "string", "translation", "localized"):
                val_idx = i
        if key_idx is None:
            key_idx = 0
        if val_idx is None:
            val_idx = 1 if len(rows[0]) > 1 else None
        if val_idx is None:
            return None
        out = {}
        ok = 0
        for r in rows[1:]:
            if len(r) <= max(key_idx, val_idx):
                continue
            k = r[key_idx].strip()
            v = r[val_idx]
            if k:
                out[k] = v
                ok += 1
        return out if ok >= 10 else None
    except Exception:
        return None

def parse_json_dict(text: str) -> dict[str, str] | None:
    s = text.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return None
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            # only keep str values
            out = {}
            ok = 0
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, str):
                    out[k] = v
                    ok += 1
            return out if ok >= 10 else None
        return None
    except Exception:
        return None

def parse_textasset_payload(payload: bytes) -> dict[str, str] | None:
    # Try utf-8, then utf-16-le.
    for enc in ("utf-8-sig", "utf-8", "utf-16-le"):
        try:
            text = payload.decode(enc, errors="strict")
            break
        except Exception:
            text = None
    if text is None:
        # fallback
        text = payload.decode("utf-8", errors="ignore")

    parsers = (parse_json_dict, parse_csv_tsv, parse_kv_lines)
    for p in parsers:
        out = p(text)
        if out:
            return out
    return None

# ---- Bundle loader ----
def load_bundle_text_tables(bundle_path: Path) -> list[dict[str, str]]:
    """
    Returns a list of dicts extracted from TextAssets in a Unity bundle.
    """
    try:
        import UnityPy  # type: ignore
    except Exception as e:
        raise SystemExit(
            "[ERR] UnityPy is required for reliable extraction.\n"
            "Install with: pip install UnityPy\n"
            f"Details: {e}"
        )

    env = UnityPy.load(str(bundle_path))
    tables: list[dict[str, str]] = []
    for obj in env.objects:
        try:
            if obj.type.name != "TextAsset":
                continue
            data = obj.read()
            payload = getattr(data, "script", None)
            if payload is None:
                continue
            if isinstance(payload, str):
                payload_b = payload.encode("utf-8", errors="ignore")
            else:
                payload_b = payload  # bytes
            table = parse_textasset_payload(payload_b)
            if table and len(table) >= 10:
                tables.append(table)
        except Exception:
            continue
    return tables

# ---- Name heuristics ----
NAME_KEY_HINT = re.compile(r"(?:^|[_\-.])(name|chara|character|speaker)(?:$|[_\-.])", re.I)

def looks_like_name_key(key: str) -> bool:
    return bool(NAME_KEY_HINT.search(key))

def looks_like_name_value(val: str) -> bool:
    v = val.strip()
    if not v:
        return False
    # short-ish, no line breaks
    if "\n" in v or "\r" in v:
        return False
    # Many character names are short (JP <= 12, EN <= 20)
    if len(v) > 30:
        return False
    # exclude obviously sentence-like strings
    if v.count(" ") >= 4:
        return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mes_root", required=True, help="MES root folder containing many subfolders")
    ap.add_argument("--people_dict", required=True, help="YAML mapping jp->kr (your existing people dict)")
    ap.add_argument("--out_csv", default="name_map.csv", help="output CSV path (utf-8-sig)")
    ap.add_argument("--out_yaml", default="people_dict_extracted.yaml", help="output YAML path (jp->kr)")
    ap.add_argument("--include_untranslated", action="store_true", help="include rows where jp not found in people_dict (kr blank)")
    args = ap.parse_args()

    mes_root = Path(args.mes_root)
    people = load_simple_yaml_mapping(Path(args.people_dict))
    jp_names = set(people.keys())

    rows = []
    # aggregate duplicates: (jp,en) -> info
    agg = defaultdict(lambda: {"folders": set(), "keys": set(), "kr": ""})

    for root, dirs, files in os.walk(mes_root):
        root_p = Path(root)
        if "lang.ja" in files and "lang.en" in files:
            ja_path = root_p / "lang.ja"
            en_path = root_p / "lang.en"

            ja_tables = load_bundle_text_tables(ja_path)
            en_tables = load_bundle_text_tables(en_path)

            # Try align by keys in each table pair; pick best overlap pairs.
            for ja_t in ja_tables:
                for en_t in en_tables:
                    common = set(ja_t.keys()) & set(en_t.keys())
                    if len(common) < 20:
                        continue
                    for k in common:
                        ja_v = str(ja_t.get(k, "")).strip()
                        en_v = str(en_t.get(k, "")).strip()
                        if not ja_v or not en_v:
                            continue

                        is_name = (ja_v in jp_names) or (looks_like_name_key(k) and looks_like_name_value(ja_v) and looks_like_name_value(en_v))
                        if not is_name:
                            continue

                        kr = people.get(ja_v, "")
                        if (not kr) and (not args.include_untranslated) and (ja_v not in jp_names):
                            # If we don't want unknown names, skip unless it's in dict.
                            continue

                        key = (ja_v, en_v)
                        agg[key]["folders"].add(str(root_p))
                        agg[key]["keys"].add(k)
                        if not agg[key]["kr"] and kr:
                            agg[key]["kr"] = kr

    # Emit CSV rows
    for (ja_v, en_v), info in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1])):
        kr = info["kr"]
        rows.append({
            "jp": ja_v,
            "en": en_v,
            "kr": kr,
            "folders": " | ".join(sorted(info["folders"])),
            "keys": " | ".join(sorted(info["keys"]))[:2000],  # avoid too huge cells
        })

    out_csv = Path(args.out_csv)
    out_yaml = Path(args.out_yaml)

    # Write CSV (Excel-friendly)
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["jp", "en", "kr", "folders", "keys"])
        w.writeheader()
        w.writerows(rows)

    # Write YAML mapping (only non-empty kr)
    with open(out_yaml, "w", encoding="utf-8", newline="") as f:
        for r in rows:
            if r["kr"]:
                # quote if needed
                jp = r["jp"].replace('"', '\\"')
                kr = r["kr"].replace('"', '\\"')
                f.write(f"\"{jp}\": \"{kr}\"\n")

    print("[OK] Done.")
    print(f"  found pairs: {len(rows)}")
    print(f"  csv: {out_csv.resolve()}")
    print(f"  yaml (only jp with kr): {out_yaml.resolve()}")

if __name__ == "__main__":
    main()
