#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract JP/EN character-name pairs from Monochrome Mobius lang.ja / lang.en UnityFS bundles.

Approach (robust / no external deps):
1) Parse UnityFS header.
2) Read compressed blocks-info blob (tries both "after header" and "at end").
3) Decompress blocks-info (supports UnityFS "raw prefix + LZ4 block" variant).
4) Parse blocks-info (supports observed BE layout with 16/32-byte hash and u16/u32 blockCount variants).
5) Decompress data blocks (LZ4 raw; tries to auto-detect small prefix before LZ4 stream; falls back to lenient decode).
6) From decompressed bytes, find dialog Labels like `00_00100_0070` and take the first speaker name string starting with '@'
   within a small window after each label.
7) Join JP and EN by label, then output unique jp_name/en_name rows with counts.

Output CSV is UTF-8 with BOM (utf-8-sig) for Excel.
"""
import argparse
import csv
import os
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


LABEL_RE = re.compile(rb'00_\d{5}_\d{4}\x00')


def read_cstr(f) -> str:
    b = bytearray()
    while True:
        c = f.read(1)
        if not c or c == b"\x00":
            break
        b.extend(c)
    return b.decode("utf-8", errors="replace")


def parse_unityfs_header(path: Path) -> dict:
    with path.open("rb") as f:
        sig = f.read(8)
        if sig != b"UnityFS\x00":
            raise ValueError(f"Not UnityFS: {path} (sig={sig!r})")
        version = struct.unpack(">I", f.read(4))[0]
        unityver = read_cstr(f)
        revision = read_cstr(f)
        size = struct.unpack(">Q", f.read(8))[0]
        cinfo_size = struct.unpack(">I", f.read(4))[0]
        uinfo_size = struct.unpack(">I", f.read(4))[0]
        flags = struct.unpack(">I", f.read(4))[0]
        header_end = f.tell()
    return {
        "sig": sig, "version": version, "unityver": unityver, "revision": revision,
        "size": size, "cinfo_size": cinfo_size, "uinfo_size": uinfo_size, "flags": flags,
        "header_end": header_end,
    }


# --- LZ4 raw block decompressor (pure python) --------------------------------
def _lz4_decompress_block(data: bytes, out_size: int) -> bytes:
    i = 0
    out = bytearray()
    n = len(data)

    while i < n and len(out) < out_size:
        token = data[i]
        i += 1

        lit_len = token >> 4
        if lit_len == 15:
            while i < n:
                b = data[i]
                i += 1
                lit_len += b
                if b != 255:
                    break

        # literals
        if lit_len:
            out.extend(data[i:i + lit_len])
            i += lit_len

        if len(out) >= out_size or i >= n:
            break

        if i + 2 > n:
            break

        offset = data[i] | (data[i + 1] << 8)
        i += 2
        if offset == 0 or offset > len(out):
            raise ValueError("LZ4 bad offset")

        match_len = (token & 0x0F) + 4
        if (token & 0x0F) == 15:
            while i < n:
                b = data[i]
                i += 1
                match_len += b
                if b != 255:
                    break

        start = len(out) - offset
        for _ in range(match_len):
            out.append(out[start])
            start += 1
            # overlapping copies are naturally handled because out grows

    if len(out) != out_size:
        # don't hard-fail: Unity bundles sometimes have padding/quirks
        return bytes(out)
    return bytes(out)


def _lz4_decompress_lenient(data: bytes, max_out: int = 500000) -> bytes:
    """Decompress until EOF or until max_out; raises on obviously bad offset early."""
    i = 0
    out = bytearray()
    n = len(data)

    while i < n and len(out) < max_out:
        token = data[i]
        i += 1

        lit_len = token >> 4
        if lit_len == 15:
            while i < n:
                b = data[i]
                i += 1
                lit_len += b
                if b != 255:
                    break

        if lit_len:
            out.extend(data[i:i + lit_len])
            i += lit_len

        if i >= n:
            break

        if i + 2 > n:
            break

        offset = data[i] | (data[i + 1] << 8)
        i += 2
        if offset == 0 or offset > len(out):
            raise ValueError("LZ4 bad offset")

        match_len = (token & 0x0F) + 4
        if (token & 0x0F) == 15:
            while i < n:
                b = data[i]
                i += 1
                match_len += b
                if b != 255:
                    break

        start = len(out) - offset
        for _ in range(match_len):
            out.append(out[start])
            start += 1

    return bytes(out)


def decompress_blocks_info(cinfo: bytes, uinfo_size: int) -> bytes:
    """
    UnityFS blocks-info in this game looks like:
      [raw prefix bytes][LZ4 raw block...]
    Prefix length varies (observed 14).
    We brute-force prefix_len in 0..32 and validate output size.
    """
    for prefix_len in range(0, 33):
        if prefix_len >= len(cinfo):
            break
        try:
            dec = _lz4_decompress_block(cinfo[prefix_len:], uinfo_size - prefix_len)
        except Exception:
            continue
        if len(dec) == (uinfo_size - prefix_len):
            return cinfo[:prefix_len] + dec
    # fallback: best-effort lenient
    for prefix_len in range(0, 33):
        try:
            dec = _lz4_decompress_lenient(cinfo[prefix_len:], max_out=uinfo_size - prefix_len)
        except Exception:
            continue
        if len(dec) > 0:
            return cinfo[:prefix_len] + dec
    raise ValueError("Failed to decompress blocks-info")


def _be_u16(b: bytes, off: int) -> int:
    return struct.unpack_from(">H", b, off)[0]


def _be_u32(b: bytes, off: int) -> int:
    return struct.unpack_from(">I", b, off)[0]


def _be_u64(b: bytes, off: int) -> int:
    return struct.unpack_from(">Q", b, off)[0]


def parse_blocks_info(uinfo: bytes, file_size: int, data_start: int) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int, str]]]:
    """
    Parse blocks-info with heuristics.
    Observed in samples:
      - hash_len = 32 bytes
      - blockCount = u16 (BE)
      - each block: u32 u_size, u32 c_size, u16 flags (BE)
      - dirCount = u32 (BE)
      - dir: u64 offset, u64 size, u32 flags, cstr name (UTF-8)
    We'll try (hash_len, blockCountWidth) combinations and pick the first plausible.
    """
    candidates = []
    for hash_len in (16, 32):
        for bcw in (2, 4):
            off = hash_len
            try:
                if off + bcw > len(uinfo):
                    continue
                if bcw == 2:
                    block_count = _be_u16(uinfo, off)
                    off += 2
                else:
                    block_count = _be_u32(uinfo, off)
                    off += 4

                if not (1 <= block_count <= 4096):
                    continue

                blocks = []
                comp_sum = 0
                for _ in range(block_count):
                    if off + 4 + 4 + 2 > len(uinfo):
                        raise ValueError("uinfo too short for blocks")
                    u_size = _be_u32(uinfo, off); off += 4
                    c_size = _be_u32(uinfo, off); off += 4
                    flags = _be_u16(uinfo, off); off += 2
                    if u_size == 0 or c_size == 0:
                        raise ValueError("zero sizes")
                    comp_sum += c_size
                    blocks.append((u_size, c_size, flags))

                if off + 4 > len(uinfo):
                    raise ValueError("missing dirCount")
                dir_count = _be_u32(uinfo, off); off += 4
                if not (0 <= dir_count <= 10000):
                    raise ValueError("dirCount implausible")

                dirs = []
                for _ in range(dir_count):
                    if off + 8 + 8 + 4 > len(uinfo):
                        raise ValueError("uinfo too short for dirs")
                    doff = _be_u64(uinfo, off); off += 8
                    dsize = _be_u64(uinfo, off); off += 8
                    dflags = _be_u32(uinfo, off); off += 4
                    end = uinfo.find(b"\x00", off)
                    if end == -1:
                        end = len(uinfo)
                    name = uinfo[off:end].decode("utf-8", errors="replace")
                    off = end + 1 if end < len(uinfo) else end
                    dirs.append((doff, dsize, dflags, name))

                # plausibility checks vs file size
                if data_start + comp_sum > file_size + 64:
                    continue
                candidates.append((hash_len, bcw, blocks, dirs))
            except Exception:
                continue

    if not candidates:
        raise ValueError("Failed to parse blocks-info structure")

    # prefer hash_len=32, bcw=2 (matches samples)
    candidates.sort(key=lambda x: (0 if (x[0], x[1]) == (32, 2) else 1))
    _, _, blocks, dirs = candidates[0]
    return blocks, dirs


def unityfs_decompress_all_blocks(path: Path) -> bytes:
    data = path.read_bytes()
    meta = parse_unityfs_header(path)
    file_size = len(data)

    header_end = meta["header_end"]
    cinfo_size = meta["cinfo_size"]
    uinfo_size = meta["uinfo_size"]

    # try blocks-info after header first, then at end
    cinfo_candidates = []
    cinfo_candidates.append(data[header_end:header_end + cinfo_size])
    cinfo_candidates.append(data[file_size - cinfo_size:file_size])

    uinfo = None
    data_start = None
    for idx, cinfo in enumerate(cinfo_candidates):
        try:
            u = decompress_blocks_info(cinfo, uinfo_size)
            # quick sanity: should contain CAB- or at least some ASCII
            if b"CAB-" in u or b"asset_bundle_variant" in u or b"Unity" in u:
                uinfo = u
                data_start = header_end + cinfo_size if idx == 0 else header_end  # best guess
                break
            # still accept if parse_blocks_info works
            blocks, _ = parse_blocks_info(u, file_size, header_end + cinfo_size)
            uinfo = u
            data_start = header_end + cinfo_size if idx == 0 else header_end
            break
        except Exception:
            continue

    if uinfo is None:
        raise ValueError(f"Cannot read blocks-info: {path}")

    # In UnityFS, data blocks usually start right after blocks-info if it's at beginning.
    # We'll compute it as header_end + cinfo_size when blocks-info is read from there.
    if data_start is None:
        data_start = header_end + cinfo_size

    blocks, _dirs = parse_blocks_info(uinfo, file_size, data_start)

    out = bytearray()
    cursor = data_start
    for u_size, c_size, bflags in blocks:
        comp = data[cursor:cursor + c_size]
        cursor += c_size

        # auto-detect small prefix before LZ4 stream
        dec = None
        for prefix in range(0, 33):
            try:
                dec_try = _lz4_decompress_lenient(comp[prefix:], max_out=max(u_size, 1024))
            except Exception:
                continue
            if len(dec_try) >= 256:
                dec = dec_try
                break
        if dec is None:
            # last resort: try strict with expected size
            for prefix in range(0, 33):
                try:
                    dec_try = _lz4_decompress_block(comp[prefix:], u_size)
                except Exception:
                    continue
                if len(dec_try) > 0:
                    dec = dec_try
                    break

        if dec is None:
            raise ValueError(f"Failed to decompress LZ4 block in {path}")

        out.extend(dec)

    return bytes(out)


def extract_utf8_cstrings(data: bytes, start: int, end: int) -> List[Tuple[int, str]]:
    """Extract UTF-8 null-terminated strings within [start, end)."""
    res = []
    i = start
    while i < end:
        if data[i] == 0:
            i += 1
            continue
        j = data.find(b"\x00", i, min(end, i + 300))
        if j == -1:
            i += 1
            continue
        chunk = data[i:j]
        try:
            s = chunk.decode("utf-8")
        except Exception:
            i += 1
            continue

        if s and all((c >= " " or c == "\t") for c in s):
            # keep strings that have letters or CJK
            if any(c.isalpha() or ("\u3040" <= c <= "\u30ff") or ("\u4e00" <= c <= "\u9fff") for c in s):
                res.append((i, s))
                i = j + 1
                continue
        i = j + 1
    return res


def normalize_name(name: Optional[str]) -> Optional[str]:
    if not name or not name.startswith("@"):
        return name
    s = name
    # strip trailing ASCII junk (often caused by broken decoding)
    while len(s) > 1 and ord(s[-1]) < 128:
        s = s[:-1]
    return s


def extract_speaker_by_label(data: bytes, window: int = 900) -> Dict[str, str]:
    """Return {label: speakerName} extracted from this lang bundle."""
    labels = [(m.start(), data[m.start():m.end()-1].decode("ascii")) for m in LABEL_RE.finditer(data)]
    out: Dict[str, str] = {}
    for idx, (pos, lab) in enumerate(labels):
        next_pos = labels[idx + 1][0] if idx + 1 < len(labels) else len(data)
        end = min(len(data), pos + window, next_pos)
        strs = extract_utf8_cstrings(data, pos, end)
        speaker = None
        for _, s in strs:
            if s.startswith("@") and len(s) > 1 and not s.startswith("@??"):
                speaker = normalize_name(s)
                break
        if speaker:
            out[lab] = speaker
    return out


def walk_lang_pairs(mes_root: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for root, _dirs, files in os.walk(mes_root):
        files_set = set(files)
        if "lang.ja" in files_set and "lang.en" in files_set:
            pairs.append((Path(root) / "lang.ja", Path(root) / "lang.en"))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mes_root", required=True, help=".../adv/mes root folder")
    ap.add_argument("--out_csv", required=True, help="output csv path")
    ap.add_argument("--also-key-hints", action="store_true", help="add examples column (folder:label hints)")
    ap.add_argument("--max_folders", type=int, default=0, help="0 = no limit (debug)")
    args = ap.parse_args()

    mes_root = Path(args.mes_root)
    out_csv = Path(args.out_csv)

    pairs = walk_lang_pairs(mes_root)
    if args.max_folders and args.max_folders > 0:
        pairs = pairs[:args.max_folders]

    total_pairs = 0
    pair_counter: Counter = Counter()
    # keep example label list for each pair
    examples: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for ja_path, en_path in pairs:
        try:
            ja_dec = unityfs_decompress_all_blocks(ja_path)
            en_dec = unityfs_decompress_all_blocks(en_path)
        except Exception as e:
            print(f"[WARN] skip {ja_path.parent}: {e}")
            continue

        ja_speakers = extract_speaker_by_label(ja_dec)
        en_speakers = extract_speaker_by_label(en_dec)

        # join on label
        common_labels = set(ja_speakers.keys()) & set(en_speakers.keys())
        for lab in common_labels:
            jp = ja_speakers.get(lab)
            en = en_speakers.get(lab)
            if not jp or not en:
                continue
            total_pairs += 1
            pair_counter[(jp, en)] += 1
            if len(examples[(jp, en)]) < 5:
                # store short hint: folder name + label
                examples[(jp, en)].append(f"{ja_path.parent.name}:{lab}")

    # write CSV (utf-8-sig for Excel)
    rows = []
    for (jp, en), cnt in sorted(pair_counter.items(), key=lambda x: (-x[1], x[0][0])):
        rows.append({
            "jp_name": jp,
            "en_name": en,
            "count": cnt,
            "examples": " | ".join(examples[(jp, en)]),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        if args.also_key_hints:
            fieldnames = ["jp_name", "en_name", "count", "examples"]
        else:
            fieldnames = ["jp_name", "en_name", "count"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        if args.also_key_hints:
            w.writerows(rows)
        else:
            w.writerows([{k:v for k,v in r.items() if k!="examples"} for r in rows])

    print("[OK] Done.")
    print(f"  scanned folder pairs (lang.ja+lang.en): {len(pairs)}")
    print(f"  extracted label-joined pairs: {total_pairs}")
    print(f"  unique jp/en name pairs: {len(rows)}")
    print(f"  csv: {out_csv}")


if __name__ == "__main__":
    main()
