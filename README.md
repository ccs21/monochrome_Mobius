# monochrome_Mobius
모노크롬 뫼비우스 한글패치 작업을 시작 합니다.

UI, 아이템 등 대사 이외의 번역은 유니티 번역 플러그인을 이용합니다.

대사는 파일 패치 형식으로 작업 합니다.

번역은 gpt 번역, 검수 및 수정은 하지 않습니다.


# GPT 번역 자동화 실행 방법

## 스캔
python main.py scan --xlsx "번역.xlsx" --dict-people "people_dict_from_excel.yaml" --dict-terms "terms_dict_from_excel_updated.yaml"

## 번역
python main.py translate --xlsx "번역.xlsx" --out "번역_ko_filled.xlsx" ^
  --env-file "env 파일 경로" ^
  --dict-people "people_dict_from_excel.yaml" ^
  --dict-terms "terms_dict_from_excel_updated.yaml" ^
  --dict-relations "relations_rules_from_excel.yaml" ^
  --dict-style "style_rules.yaml"
