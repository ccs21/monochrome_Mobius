\
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import openpyxl
import yaml
from tqdm import tqdm

try:
    from dotenv import dotenv_values
except Exception:
    dotenv_values = None

# OpenAI SDK (Responses API recommended)
from openai import OpenAI  # type: ignore


# -----------------------------
# Utils
# -----------------------------

TAG_RE = re.compile(r"\{[^{}]*\}")
RUBY_RE = re.compile(r"\[([^*\]]+)\*([^\]]+)\]")

# Characters that often appear as extraction noise in this project
NOISE_CHARS = set(["Ц", "Ч"])


def read_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_excel_text(s: str) -> str:
    # Excel sometimes contains _x000D_\n sequences
    return (s or "").replace("_x000D_\n", "\n").replace("\r\n", "\n").replace("\r", "\n")


def remove_noise_outside_tags(s: str) -> str:
    # Remove known garbage chars, but keep anything inside { ... } intact.
    if not s:
        return s
    parts = []
    last = 0
    for m in TAG_RE.finditer(s):
        before = s[last:m.start()]
        before = "".join(ch for ch in before if ch not in NOISE_CHARS)
        parts.append(before)
        parts.append(m.group(0))
        last = m.end()
    tail = s[last:]
    tail = "".join(ch for ch in tail if ch not in NOISE_CHARS)
    parts.append(tail)
    return "".join(parts)


def strip_ruby_notation(s: str) -> str:
    # [兄*あに] -> 兄
    return RUBY_RE.sub(lambda m: m.group(1), s or "")


def stable_hash(*items: str) -> str:
    h = sha256()
    for it in items:
        h.update(it.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


# -----------------------------
# Cache
# -----------------------------

class SqliteCache:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT NOT NULL, created_at INTEGER NOT NULL)"
        )
        self.conn.commit()

    def get(self, k: str) -> Optional[str]:
        cur = self.conn.execute("SELECT v FROM cache WHERE k=?", (k,))
        row = cur.fetchone()
        return row[0] if row else None

    def set(self, k: str, v: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO cache (k, v, created_at) VALUES (?, ?, ?)",
            (k, v, int(time.time())),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


# -----------------------------
# Dictionary bundle
# -----------------------------

@dataclass(frozen=True)
class DictBundle:
    people: Dict[str, dict]
    terms: Dict[str, str]
    relations: dict
    style: dict

    def signature(self) -> str:
        # used for caching. Keep short and deterministic.
        people_sig = stable_hash(json.dumps(self.people, ensure_ascii=False, sort_keys=True))
        terms_sig = stable_hash(json.dumps(self.terms, ensure_ascii=False, sort_keys=True))
        rel_sig = stable_hash(json.dumps(self.relations, ensure_ascii=False, sort_keys=True))
        style_sig = stable_hash(json.dumps(self.style, ensure_ascii=False, sort_keys=True))
        return stable_hash(people_sig, terms_sig, rel_sig, style_sig)


def load_dicts(people_path: str, terms_path: str, relations_path: str | None, style_path: str | None) -> DictBundle:
    people = read_yaml(people_path)
    terms = read_yaml(terms_path)
    relations = read_yaml(relations_path) if relations_path else {}
    style = read_yaml(style_path) if style_path else {}
    return DictBundle(people=people, terms=terms, relations=relations, style=style)


# -----------------------------
# Excel IO
# -----------------------------

def load_master_dialog_rows(xlsx_path: str) -> Tuple[openpyxl.Workbook, openpyxl.worksheet.worksheet.Worksheet, List[dict]]:
    wb = openpyxl.load_workbook(xlsx_path)
    if "master_dialog" not in wb.sheetnames:
        raise ValueError("Sheet 'master_dialog' not found")
    ws = wb["master_dialog"]

    # Expect header row at row 1
    headers = [c.value for c in ws[1]]
    expected = ["scenario_dir", "Label", "CharacterName", "ja", "ko"]
    for e in expected:
        if e not in headers:
            raise ValueError(f"Missing column: {e}. Found: {headers}")

    idx = {h: headers.index(h) + 1 for h in headers}

    rows = []
    for r in range(2, ws.max_row + 1):
        row = {
            "rownum": r,
            "scenario_dir": ws.cell(r, idx["scenario_dir"]).value,
            "Label": ws.cell(r, idx["Label"]).value,
            "CharacterName": ws.cell(r, idx["CharacterName"]).value,
            "ja": normalize_excel_text(ws.cell(r, idx["ja"]).value or ""),
            "ko": normalize_excel_text(ws.cell(r, idx["ko"]).value or ""),
        }
        rows.append(row)

    return wb, ws, rows


def write_ko(ws, rows: List[dict], ko_map: Dict[int, str]) -> None:
    # Find ko column
    headers = [c.value for c in ws[1]]
    ko_col = headers.index("ko") + 1
    for row in rows:
        r = row["rownum"]
        if r in ko_map:
            ws.cell(r, ko_col).value = ko_map[r]


# -----------------------------
# Scanning
# -----------------------------

KATAKANA_WORD_RE = re.compile(r"[ァ-ヴー]{3,}")  # rough: >=3 katakana chars
KANJI_TERM_RE = re.compile(r"[一-龥]{2,}")      # rough: >=2 kanji chars


def scan_unknown_speakers(rows: List[dict], people_dict: Dict[str, dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        spk = (row.get("CharacterName") or "").strip()
        if spk and spk.startswith("@") and spk not in people_dict:
            counts[spk] = counts.get(spk, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))


def scan_term_candidates(rows: List[dict], terms_dict: Dict[str, str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        s = row.get("ja") or ""
        # Ignore tags; only scan outside tags
        s_no_tags = TAG_RE.sub(" ", s)
        for m in KATAKANA_WORD_RE.finditer(s_no_tags):
            w = m.group(0)
            if w not in terms_dict:
                counts[w] = counts.get(w, 0) + 1
        # Also scan kanji terms (optional)
        for m in KANJI_TERM_RE.finditer(s_no_tags):
            w = m.group(0)
            if w not in terms_dict:
                # too many false positives; keep only repeated ones later
                counts[w] = counts.get(w, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))


# -----------------------------
# OpenAI translation
# -----------------------------

def load_api_key(env_file: str) -> str:
    p = Path(env_file)
    if not p.exists():
        raise FileNotFoundError(env_file)

    if dotenv_values is not None:
        vals = dotenv_values(env_file)
        key = vals.get("OPENAI_API_KEY")
        if key:
            return key.strip()

    # Fallback: manual parse
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise ValueError("OPENAI_API_KEY not found in env file")


def build_system_instructions(bundle: DictBundle) -> str:
    # This is intentionally concise. Keep large dictionaries out of the system prompt
    # and pass them as separate sections to reduce accidental drift.
    # You can swap this with your own `GPT 번역 프롬프트.txt` content if you want.
    return (
        "너는 '칭송받는 자 시리즈: 모노크롬 뫼비우스' 한국어 현지화 번역가다.\n"
        "목표: 게임에 실제로 넣어도 어색하지 않은 자연스러운 번역.\n"
        "규칙:\n"
        "1) 출력은 반드시 JSON 배열. 입력 배열과 길이 동일.\n"
        "2) 각 요소는 반드시 큰따옴표로 감싼 문자열 형태(일반 문자열)이며, 추가 텍스트 금지.\n"
        "3) 원문 안의 { ... } 태그는 문자 단위로 절대 수정/삭제/추가/이동하지 말 것.\n"
        "4) [漢字*ふりがな] 형태는 출력에서 루비를 제거하고 漢字만 남길 것.\n"
        "5) 원문에 섞인 추출 잡음(예: Ц, Ч)은 { ... } 밖에서는 출력에서 제거.\n"
        "6) 영어는 번역하지 말고 그대로 유지.\n"
        "7) 용어/인물/관계 규칙은 아래 사전 섹션을 반드시 준수.\n"
    )


def build_dict_section(bundle: DictBundle) -> str:
    # Keep this deterministic to improve caching.
    people_lines = []
    for k in sorted(bundle.people.keys()):
        v = bundle.people[k]
        # v may be mapping or string
        if isinstance(v, dict):
            ko = v.get("ko") or v.get("name_ko") or v.get("name") or ""
            gender = v.get("gender") or v.get("sex") or ""
        else:
            ko = str(v)
            gender = ""
        if ko:
            people_lines.append(f"{k}={ko}/{gender}".strip("/"))
    terms_lines = [f"{k}={bundle.terms[k]}" for k in sorted(bundle.terms.keys())]

    rel_lines = []
    # relations file is expected to be simple rules; keep as yaml block
    if bundle.relations:
        rel_lines.append(yaml.safe_dump(bundle.relations, allow_unicode=True, sort_keys=True).strip())
    style_lines = []
    if bundle.style:
        style_lines.append(yaml.safe_dump(bundle.style, allow_unicode=True, sort_keys=True).strip())

    return (
        "### 인물 사전\n" + "\n".join(people_lines) + "\n\n"
        "### 용어 사전\n" + "\n".join(terms_lines) + "\n\n"
        "### 관계/호칭 규칙\n" + ("\n".join(rel_lines) if rel_lines else "(없음)") + "\n\n"
        "### 문체 규칙\n" + ("\n".join(style_lines) if style_lines else "(없음)") + "\n"
    )


def validate_tag_preservation(src_list: List[str], out_list: List[str]) -> Tuple[bool, str]:
    for i, (src, out) in enumerate(zip(src_list, out_list)):
        src_tags = TAG_RE.findall(src)
        out_tags = TAG_RE.findall(out)
        if src_tags != out_tags:
            return False, f"tag_mismatch at index {i}: src_tags={src_tags} out_tags={out_tags}"
    return True, ""


def translate_batch(
    client: OpenAI,
    model: str,
    bundle: DictBundle,
    src_lines: List[str],
    temperature: float = 0.2,
    max_output_tokens: int = 4000,
) -> List[str]:
    sys_inst = build_system_instructions(bundle)
    dict_block = build_dict_section(bundle)

    # The actual request
    user_input = {
        "task": "Translate JA->KO for game localization.",
        "notes": "Return JSON array only.",
        "dicts": dict_block,
        "lines": src_lines,
    }

    # Use Responses API
    resp = client.responses.create(
        model=model,
        instructions=sys_inst,
        input=json.dumps(user_input, ensure_ascii=False),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    text = resp.output_text.strip()

    # Strict JSON parse
    try:
        data = json.loads(text)
    except Exception:
        # Try to salvage if model wrapped with code fences
        text2 = text
        text2 = text2.strip("` \n")
        data = json.loads(text2)

    if not isinstance(data, list) or len(data) != len(src_lines):
        raise ValueError(f"Bad output shape. expected list[{len(src_lines)}], got {type(data)} len={len(data) if isinstance(data, list) else 'n/a'}")

    # Post-processing safety: ruby removal & noise removal outside tags
    out_lines: List[str] = []
    for s in data:
        if not isinstance(s, str):
            s = str(s)
        s = strip_ruby_notation(s)
        s = remove_noise_outside_tags(s)
        out_lines.append(s)

    ok, msg = validate_tag_preservation(src_lines, out_lines)
    if not ok:
        raise ValueError(msg)

    return out_lines


def chunked(seq: List[dict], n: int) -> List[List[dict]]:
    return [seq[i:i+n] for i in range(0, len(seq), n)]


def is_needs_translation(row: dict) -> bool:
    ko = (row.get("ko") or "").strip()
    ja = (row.get("ja") or "").strip()
    if not ja:
        return False
    # Translate if ko is empty OR ko identical to ja OR ko contains obvious leftover JP (katakana/kanji)
    if not ko:
        return True
    if ko == ja:
        return True
    if re.search(r"[ぁ-んァ-ヴ一-龥]", ko):
        # If ko still contains Japanese, treat as needing re-translation
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_scan = sub.add_parser("scan", help="scan missing speakers/terms in the excel")
    ap_scan.add_argument("--xlsx", required=True)
    ap_scan.add_argument("--dict-people", required=True)
    ap_scan.add_argument("--dict-terms", required=True)
    ap_scan.add_argument("--top", type=int, default=200)

    ap_tr = sub.add_parser("translate", help="translate ja->ko and write out new xlsx")
    ap_tr.add_argument("--xlsx", required=True)
    ap_tr.add_argument("--out", required=True)
    ap_tr.add_argument("--env-file", required=True)
    ap_tr.add_argument("--model", default="gpt-4.1")
    ap_tr.add_argument("--chunk", type=int, default=40)
    ap_tr.add_argument("--dict-people", required=True)
    ap_tr.add_argument("--dict-terms", required=True)
    ap_tr.add_argument("--dict-relations", default=None)
    ap_tr.add_argument("--dict-style", default=None)
    ap_tr.add_argument("--cache-db", default=".cache.sqlite3")
    ap_tr.add_argument("--max-retries", type=int, default=2)

    args = ap.parse_args()

    if args.cmd == "scan":
        wb, ws, rows = load_master_dialog_rows(args.xlsx)
        people = read_yaml(args.dict_people)
        terms = read_yaml(args.dict_terms)
        unk_spk = scan_unknown_speakers(rows, people)
        cand_terms = scan_term_candidates(rows, terms)

        print("[unknown_speakers]")
        for k, v in list(unk_spk.items())[: args.top]:
            print(f"{k}\t{v}")
        print("\n[term_candidates]")
        for k, v in list(cand_terms.items())[: args.top]:
            print(f"{k}\t{v}")
        return

    if args.cmd == "translate":
        key = load_api_key(args.env_file)
        client = OpenAI(api_key=key)

        bundle = load_dicts(args.dict_people, args.dict_terms, args.dict_relations, args.dict_style)
        wb, ws, rows = load_master_dialog_rows(args.xlsx)

        to_tr = [r for r in rows if is_needs_translation(r)]
        if not to_tr:
            print("Nothing to translate.")
            wb.save(args.out)
            return

        cache = SqliteCache(Path(args.cache_db))
        ko_map: Dict[int, str] = {}

        sig = bundle.signature()
        pbar = tqdm(total=len(to_tr), desc="Translating", unit="line")

        for group in chunked(to_tr, args.chunk):
            src_lines = [r["ja"] for r in group]
            cache_key = stable_hash("v1", args.model, sig, json.dumps(src_lines, ensure_ascii=False))
            cached = cache.get(cache_key)
            if cached:
                out_lines = json.loads(cached)
                for r, out in zip(group, out_lines):
                    ko_map[r["rownum"]] = out
                pbar.update(len(group))
                continue

            last_err = None
            for attempt in range(args.max_retries + 1):
                try:
                    out_lines = translate_batch(client, args.model, bundle, src_lines)
                    # Save cache
                    cache.set(cache_key, json.dumps(out_lines, ensure_ascii=False))
                    for r, out in zip(group, out_lines):
                        ko_map[r["rownum"]] = out
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    # backoff
                    time.sleep(1.0 + attempt * 1.5)

            if last_err is not None:
                cache.close()
                raise RuntimeError(f"Failed to translate batch starting at row {group[0]['rownum']}: {last_err}") from last_err

            pbar.update(len(group))

        pbar.close()

        write_ko(ws, rows, ko_map)
        wb.save(args.out)
        cache.close()
        print(f"Saved: {args.out}")
        return


if __name__ == "__main__":
    main()
