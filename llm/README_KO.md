# MixSense AI Search — 로컬 실행 가이드 (GPT Intent Parser 포함)

이 프로젝트는 **LLM을 추천자가 아니라 Intent Parser(자연어 → JSON 조건)**로만 쓰고,
실제 결과는 **카탈로그 기반 엔진**이 결정합니다.

## 1) 폴더 구조 (압축 해제 후)

```
mixsense_ai_search_project/
  src/                    # 파이썬 패키지 코드
  mixsense_outputs/        # 준비된 데이터(카탈로그/allowed values 등)
  run_local.py             # ✅ 제일 쉬운 실행 파일(인터랙티브)
  requirements.txt
  .env.example
```

## 2) 파이썬 환경 준비 (권장: Python 3.10~3.12)

### macOS / Linux
```bash
cd mixsense_ai_search_project
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
cd mixsense_ai_search_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

## 3) OpenAI API Key 설정 (GPT 파서 사용)

1) OpenAI 대시보드에서 API Key를 만든 다음
2) 환경변수 `OPENAI_API_KEY`로 설정하거나 `.env` 파일에 넣습니다.

### 방법 A: .env 파일(추천)
```bash
cp .env.example .env
# .env 열어서 OPENAI_API_KEY 값을 채우세요
```

### 방법 B: 환경변수
macOS / Linux:
```bash
export OPENAI_API_KEY="your_api_key_here"
```

Windows PowerShell:
```powershell
setx OPENAI_API_KEY "your_api_key_here"
```

## 4) 실행 (가장 쉬운 방법)
```bash
python run_local.py
```

실행 후, 예를 들어 아래처럼 물어보면 됩니다:
- `오늘 같이 햇빛 좋은 날 듣기 좋은 노래 추천해줘`
- `비 오는 날 드라이브 R&B 잔잔한`
- `운동할 때 신나는 힙합`

## 5) CLI로 1회 실행(옵션)
```bash
PYTHONPATH=./src python -m mixsense_ai_search.cli \
  --catalog ./mixsense_outputs/prepared_catalog.pkl \
  --allowed ./mixsense_outputs/taxonomy_allowed_values.json \
  --parser gpt \
  --query "오늘 같이 햇빛 좋은 날 듣기 좋은 노래 추천해줘" \
  --k 10
```

## 6) 자주 터지는 문제 2개

### (1) OPENAI_API_KEY 관련 에러
- `.env` 또는 환경변수에 `OPENAI_API_KEY`가 설정되어야 합니다.

### (2) 결과가 이상하거나 0건이 자주 나옴
- v1은 `mood_tags/context_tags`를 **필터가 아니라 랭킹 보너스**로 씁니다.
- 그래도 태그가 희귀하면 검색이 불안정해집니다.
  → 다음 단계(1번)에서 CORE taxonomy 압축을 적용하면 안정성이 올라갑니다.


## 결과 개수 기본값(UX 규칙)

- 추천/탐색 질의: 기본 5개
- "이 노래 뭐야?" 식별 질의: TOP3 (항상 3개 이하)
- 사용자가 "1곡"처럼 명시하면 그 수를 반영하되, 식별 질의는 최대 3개로 제한합니다.
