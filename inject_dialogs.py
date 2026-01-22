import argparse
from pathlib import Path
import csv
import re
from collections import Counter, defaultdict

import UnityPy

TOKEN_RE = re.compile(r"\{[^{}]+\}")  # {d000700} 같은 토큰 전부

def tokens_counter(s: str) -> Counter:
    return Counter(TOKEN_RE.findall(s or ""))

def validate_rows(rows, error_report_path: Path):
    """
    FAIL FAST: en 토큰과 ko 토큰이 정확히 일치(개수까지)해야 통과.
    (원문 변경 금지, 프리픽스 누락 방지 목적)
    """
    errors = []
    for r in rows:
        scenario = r["scenario_dir"]
        label = r["Label"]
        en = r["en"] or ""
        ko = r["ko"] or ""

        en_tok = tokens_counter(en)
        ko_tok = tokens_counter(ko)

        if en_tok != ko_tok:
            # 어떤 토큰이 빠졌는지/추가됐는지 구체화
            missing = list((en_tok - ko_tok).elements())
            extra = list((ko_tok - en_tok).elements())
            errors.append({
                "scenario_dir": scenario,
                "Label": label,
                "missing_tokens": " ".join(missing),
                "extra_tokens": " ".join(extra),
                "en": en,
                "ko": ko,
            })
            # FAIL FAST: 하나 발견되면 바로 끝내도 되지만, 한 번에 다 고치기 편하게 전부 모아 출력
    if errors:
        error_report_path.parent.mkdir(parents=True, exist_ok=True)
        with error_report_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["scenario_dir","Label","missing_tokens","extra_tokens","en","ko"])
            w.writeheader()
            w.writerows(errors)
        print(f"[FAIL] Token validation failed: {len(errors)} issue(s)")
        print(f"[FAIL] Report written: {error_report_path}")
        raise SystemExit(2)

def load_env(bundle_path: Path):
    return UnityPy.load(str(bundle_path))

def save_single_file_env(env, out_path: Path):
    # lang.en 같은 단일 번들은 보통 env.files가 1개
    if len(env.files) != 1:
        raise SystemExit(f"[ERR] unexpected file count in bundle ({len(env.files)}): {out_path}")
    file = next(iter(env.files.values()))
    out_path.write_bytes(file.save())

def inject_one_lang_en(lang_en_path: Path, label_to_ko: dict) -> bytes:
    env = load_env(lang_en_path)
    changed = 0
    total = 0
    for obj in env.objects:
        if obj.type.name != "MonoBehaviour":
            continue
        tree = obj.read_typetree()
        param = tree.get("param")
        if not isinstance(param, list):
            continue

        for entry in param:
            total += 1
            lbl = entry.get("Label", "")
            if lbl in label_to_ko:
                entry["messageText"] = label_to_ko[lbl]
                changed += 1

        obj.save_typetree(tree)

    # 저장 바이트 반환
    if len(env.files) != 1:
        raise SystemExit(f"[ERR] unexpected file count in bundle ({len(env.files)}): {lang_en_path}")
    file = next(iter(env.files.values()))
    data = file.save()

    return data, changed, total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mes_root", required=True, help="adv/mes 폴더")
    ap.add_argument("--master_csv", required=True, help="export로 만든 마스터 CSV (ko 채워진 버전)")
    ap.add_argument("--files_out_dir", required=True, help=r'배포용 폴더 (예: F:\모노크롬 한글패치\files)')
    ap.add_argument("--error_report", default="inject_errors.csv", help="검증 실패 리포트 파일명")
    ap.add_argument("--dry_run", action="store_true", help="검증만 하고 실제 주입/저장은 하지 않음")
    args = ap.parse_args()

    mes_root = Path(args.mes_root)
    master_csv = Path(args.master_csv)
    files_out = Path(args.files_out_dir)
    error_report_path = Path(args.error_report)
    if not error_report_path.is_absolute():
        error_report_path = master_csv.parent / error_report_path

    if not master_csv.exists():
        raise SystemExit(f"[ERR] master_csv not found: {master_csv}")

    # 1) CSV 로드 + ko 있는 행만 대상
    rows = []
    with master_csv.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        required = {"scenario_dir","Label","en","ko"}
        if not required.issubset(set(r.fieldnames or [])):
            raise SystemExit(f"[ERR] CSV header missing. Need at least: {sorted(required)}")
        for row in r:
            ko = (row.get("ko") or "").strip()
            if ko:
                rows.append({
                    "scenario_dir": (row.get("scenario_dir") or "").strip(),
                    "Label": (row.get("Label") or "").strip(),
                    "CharacterName": row.get("CharacterName",""),
                    "ja": row.get("ja",""),
                    "en": row.get("en",""),
                    "ko": ko,
                })

    if not rows:
        raise SystemExit("[ERR] 주입 대상이 없습니다. (CSV의 ko 칸이 전부 비어있음)")

    # 2) 전체 검증(FAIL FAST)
    validate_rows(rows, error_report_path)

    print(f"[OK] Validation passed. rows_to_inject={len(rows)}")

    if args.dry_run:
        print("[DRY RUN] No files were modified.")
        return

    # 3) 시나리오 폴더별로 묶어서 주입
    grouped = defaultdict(dict)  # scenario_dir -> {Label: ko}
    for r in rows:
        grouped[r["scenario_dir"]][r["Label"]] = r["ko"]

    files_out.mkdir(parents=True, exist_ok=True)

    total_scenarios = 0
    total_changed = 0
    total_entries = 0

    for scenario_dir, label_map in grouped.items():
        lang_en_path = mes_root / scenario_dir / "lang.en"
        if not lang_en_path.exists():
            raise SystemExit(f"[ERR] lang.en not found: {lang_en_path}")

        data, changed, total = inject_one_lang_en(lang_en_path, label_map)

        # (A) 원본에 바로 덮어쓰기
        lang_en_path.write_bytes(data)

        # (B) 배포용 파일 저장: scenarios_XX_YYYYY_lang.ko
        out_patch = files_out / f"{scenario_dir}_lang.ko"
        out_patch.write_bytes(data)

        total_scenarios += 1
        total_changed += changed
        total_entries += total

        print(f"[OK] {scenario_dir}: changed={changed} / entries={total}")

    print(f"[DONE] scenarios={total_scenarios} changed_lines={total_changed} total_entries_scanned={total_entries}")
    print(f"[DONE] files_out: {files_out}")

if __name__ == "__main__":
    main()
