#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hash_dump.py (v2)

원본(original) 모드:
  1) root 하위에서 'lang.en' **폴더**를 찾으면: 그 폴더들 안의 파일을 전부 수집
  2) 'lang.en' 폴더가 하나도 없으면: root 하위 파일 중에서 **파일명에 'lang.en'이 포함된 파일**을 수집 (fallback)

패치(patch) 모드:
  - 지정한 patch 폴더(기본: <root>/patch) 안의 모든 파일을 수집

출력(prefix 기본: hashes_<mode>):
  - hashes_<mode>.csv        (utf-8-sig, Excel friendly)
  - hashes_<mode>.json
  - hashes_<mode>_dict.py.txt (패쳐 하드코딩용 dict 스니펫)

예)
  python hash_dump.py --mode original --root "F:\...\adv\mes" --token "lang.en" --expect 399
  python hash_dump.py --mode patch --patch "F:\patch" --expect 399
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()

def to_posix_rel(p: Path) -> str:
    return p.as_posix()

def iter_files_under(base: Path) -> list[Path]:
    files: list[Path] = []
    for p in base.rglob("*"):
        if p.is_file():
            files.append(p)
    return files

def find_dirs_named(root: Path, name: str) -> list[Path]:
    dirs: list[Path] = []
    if root.is_dir() and root.name.lower() == name.lower():
        return [root]
    for p in root.rglob("*"):
        try:
            if p.is_dir() and p.name.lower() == name.lower():
                dirs.append(p)
        except Exception:
            continue
    # dedupe
    uniq = []
    seen = set()
    for d in dirs:
        try:
            k = str(d.resolve())
        except Exception:
            k = str(d)
        if k not in seen:
            uniq.append(d)
            seen.add(k)
    return uniq

def dump_hashes(files: list[Path], rel_base: Path, out_prefix: Path, expect: int | None = None) -> None:
    rows = []
    d: dict[str, dict] = {}

    for fp in files:
        try:
            rel = fp.relative_to(rel_base)
        except Exception:
            rel = Path(fp.name)

        rel_key = to_posix_rel(rel)
        h = sha1_file(fp)
        size = fp.stat().st_size

        rows.append({"rel_path": rel_key, "sha1": h, "size": size})
        d[rel_key] = {"sha1": h, "size": size}

    rows.sort(key=lambda r: r["rel_path"])

    if expect is not None and len(rows) != expect:
        print(f"[WARN] 파일 수가 기대값과 다릅니다: got={len(rows)} expect={expect}")

    csv_path = out_prefix.with_suffix(".csv")
    json_path = out_prefix.with_suffix(".json")
    py_path  = out_prefix.parent / (out_prefix.name + "_dict.py.txt")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["rel_path", "sha1", "size"])
        w.writeheader()
        w.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

    with py_path.open("w", encoding="utf-8") as f:
        f.write("# paste this into your patcher\n")
        f.write("HASHES = {\n")
        for r in rows:
            f.write(f'    "{r["rel_path"]}": {{"sha1": "{r["sha1"]}", "size": {r["size"]}}},\n')
        f.write("}\n")

    print(f"[OK] wrote: {csv_path}")
    print(f"[OK] wrote: {json_path}")
    print(f"[OK] wrote: {py_path}")
    print(f"[DONE] files={len(rows)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["original", "patch"], required=True)
    ap.add_argument("--root", default=".", help="(original 모드) 검색 루트")
    ap.add_argument("--token", default="lang.en", help="original 모드에서 찾을 폴더명/파일명 토큰 (기본: lang.en)")
    ap.add_argument("--patch", default=None, help="(patch 모드) patch 폴더 경로. 기본은 <root>/patch")
    ap.add_argument("--out", default=None, help="출력 prefix(확장자 제외). 기본 hashes_<mode>")
    ap.add_argument("--expect", type=int, default=None, help="기대 파일 수(예: 399)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_prefix = Path(args.out).resolve() if args.out else (Path.cwd() / f"hashes_{args.mode}")

    if args.mode == "original":
        if not root.exists():
            raise SystemExit(f"[ERR] root not found: {root}")

        # 1) 폴더 탐색
        lang_dirs = find_dirs_named(root, args.token)

        if lang_dirs:
            all_files: list[Path] = []
            for d in lang_dirs:
                all_files.extend(iter_files_under(d))

            # dedupe
            uniq = []
            seen = set()
            for f in all_files:
                try:
                    k = str(f.resolve())
                except Exception:
                    k = str(f)
                if k not in seen:
                    uniq.append(f)
                    seen.add(k)

            print(f"[INFO] found '{args.token}' dirs: {len(lang_dirs)}")
            for d in lang_dirs:
                print(" -", d)

            dump_hashes(uniq, rel_base=root, out_prefix=out_prefix, expect=args.expect)
            return

        # 2) fallback: 파일명에 token 포함된 것만 수집
        print(f"[WARN] '{args.token}' 폴더를 찾지 못했습니다. 폴더 대신 파일명에 '{args.token}'가 포함된 파일로 fallback 합니다.")
        all_files = iter_files_under(root)
        hit = [p for p in all_files if args.token.lower() in p.name.lower()]

        if not hit:
            raise SystemExit(f"[ERR] '{args.token}' 폴더도 없고, 파일명에 '{args.token}'가 포함된 파일도 없습니다. root={root}")

        dump_hashes(hit, rel_base=root, out_prefix=out_prefix, expect=args.expect)

    else:
        patch_dir = Path(args.patch).resolve() if args.patch else (root / "patch")
        if not patch_dir.exists() or not patch_dir.is_dir():
            raise SystemExit(f"[ERR] patch dir not found: {patch_dir}")
        files = iter_files_under(patch_dir)
        dump_hashes(files, rel_base=patch_dir, out_prefix=out_prefix, expect=args.expect)

if __name__ == "__main__":
    main()
