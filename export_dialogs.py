import argparse
from pathlib import Path
import csv
import UnityPy

def load_param_list(bundle_path: Path):
    """Return list[dict] of entries from MonoBehaviour.typetree['param']."""
    if not bundle_path.exists():
        return None
    env = UnityPy.load(str(bundle_path))
    for obj in env.objects:
        if obj.type.name == "MonoBehaviour":
            tree = obj.read_typetree()
            param = tree.get("param")
            if isinstance(param, list):
                return param
    return None

def index_by_label(param_list):
    if not param_list:
        return {}
    out = {}
    for e in param_list:
        lbl = e.get("Label", "")
        if lbl:
            out[lbl] = e
    return out

def find_scenario_dirs(mes_root: Path):
    # mes_root 아래에서 scenarios_* 폴더를 모두 찾되, lang.en 또는 lang.ja가 있는 폴더만 대상
    dirs = []
    for d in mes_root.rglob("scenarios_*"):
        if not d.is_dir():
            continue
        if (d / "lang.en").exists() or (d / "lang.ja").exists():
            dirs.append(d)
    # 경로 길이/정렬 안정성: 폴더명 기준 + 경로 기준
    dirs.sort(key=lambda p: (p.name, str(p)))
    return dirs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mes_root", required=True, help="adv/mes 폴더 (scenarios_*들이 있는 위치)")
    ap.add_argument("--out_csv", required=True, help="출력 CSV 경로")
    args = ap.parse_args()

    mes_root = Path(args.mes_root)
    out_csv = Path(args.out_csv)

    scenario_dirs = find_scenario_dirs(mes_root)
    if not scenario_dirs:
        raise SystemExit(f"[ERR] scenarios_* 폴더를 찾지 못했습니다: {mes_root}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    folders = 0

    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["scenario_dir","Label","CharacterName","ja","en","ko"])

        for sd in scenario_dirs:
            folders += 1
            ja_list = load_param_list(sd / "lang.ja")
            en_list = load_param_list(sd / "lang.en")

            ja_m = index_by_label(ja_list)
            en_m = index_by_label(en_list)

            labels = sorted(set(ja_m.keys()) | set(en_m.keys()))
            for lbl in labels:
                e_ja = ja_m.get(lbl, {})
                e_en = en_m.get(lbl, {})

                char_name = e_ja.get("CharacterName") or e_en.get("CharacterName") or ""
                ja_txt = e_ja.get("messageText", "") if e_ja else ""
                en_txt = e_en.get("messageText", "") if e_en else ""

                w.writerow([sd.name, lbl, char_name, ja_txt, en_txt, ""])
                total_rows += 1

    print(f"[OK] folders={folders} rows={total_rows}")
    print(f"[OK] wrote: {out_csv}")

if __name__ == "__main__":
    main()
