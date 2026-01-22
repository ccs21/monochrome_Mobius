# Monochrome Möbius Translator (Excel -> ko)

## What it does
- Reads `번역.xlsx` (sheet: `master_dialog`) with columns:
  `scenario_dir, Label, CharacterName, ja, ko`
- Translates `ja` -> `ko` with:
  - tag preservation: `{...}` must remain *byte-identical*
  - ruby removal: `[兄*あに]` -> `兄` (output only)
  - noise removal: stray `Ц`, `Ч` outside `{...}` removed (output only)
  - glossary enforcement via YAML dictionaries
  - relationship/honorific rules (YAML)

## Two-phase workflow (recommended)
1) **Scan**: detect missing speakers/terms and show candidates.
2) **Translate**: only after dictionaries are ready.

## Setup
1. Create your env file (NOT committed):
   `F:\모노크롬 한글패치\work\.env`

   Example:
   `OPENAI_API_KEY=sk-proj-...`

2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```

## Run
### 1) Scan dictionaries
```bash
python main.py scan --xlsx "번역.xlsx" --dict-people "people_dict_from_excel.yaml" --dict-terms "terms_dict_from_excel_updated.yaml"
```

### 2) Translate
```bash
python main.py translate --xlsx "번역.xlsx" --out "번역_ko_filled.xlsx" ^
  --env-file "F:\모노크롬 한글패치\work\.env" ^
  --dict-people "people_dict_from_excel.yaml" ^
  --dict-terms "terms_dict_from_excel_updated.yaml" ^
  --dict-relations "relations_rules_from_excel.yaml" ^
  --dict-style "style_rules.yaml"
```

### Notes
- Translation uses chunking (default 40 lines/request) + SQLite cache to reduce cost.
- It *validates* tag preservation and line-count match; on failure it retries with stricter instructions.
