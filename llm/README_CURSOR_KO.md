# MixSense AI Search 실행하기

이 문서는 **Cursor(=VS Code 계열 IDE)** 기준으로, 로컬에서 `run_local.py`를 실행해
자연어 질의 → (GPT Intent Parser) → 카탈로그 추천 결과를 확인하는 방법을 정리합니다.

## 1) 폴더 열기
1. 압축 해제 후 `mixsense_ai_search_project/` 폴더를 준비합니다. (예: Desktop)
2. Cursor → **File → Open Folder...** → `mixsense_ai_search_project` 선택

## 2) Cursor 터미널 열기
- Cursor 상단 메뉴 **View → Terminal**  
  (단축키: `Ctrl + ``)

이제 아래 명령은 모두 **Cursor 터미널에서** 실행합니다.

## 3) Python 확인
```bash
python3 --version
```
- `Python 3.9+`이면 OK  
- `command not found`면 Homebrew로 설치 후 다시 확인하세요.

## 4) 가상환경 만들기 + 활성화
```bash
cd mixsense_ai_search_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

## 5) (중요) Cursor에서 Python 인터프리터 선택
Cursor에서 자동완성/실행이 시스템 Python으로 붙는 걸 막기 위해:
1. Command Palette 열기: `Cmd + Shift + P`
2. `Python: Select Interpreter` 실행
3. 목록에서 `./.venv/bin/python` 선택

만약 `Python: Select Interpreter`가 안 보이면:
- Extensions에서 **Python 확장(ms-python.python)** 설치 후 다시 시도

## 6) OpenAI API Key 설정 (.env 추천)
1) 프로젝트 루트에서 `.env` 파일 만들기
```bash
cp .env.example .env
```

2) Cursor에서 `.env` 열고 다음을 채우기:
```env
OPENAI_API_KEY="여기에_너의_API_KEY"
MIXSENSE_OPENAI_MODEL="gpt-4o-mini"
```

> ⚠️ `.env`는 절대 깃에 올리지 마세요. (이미 `.gitignore`에 포함)

## 7) 실행
```bash
python run_local.py
```

성공하면 이렇게 보입니다:
- `parser mode: gpt`  → GPT Intent Parser 사용 중
- `parser mode: rule` → API 키를 못 읽어서 룰 기반(무료)으로 실행 중

## 8) 테스트 질의 예시
터미널 프롬프트에 그대로 입력:
- `오늘 같이 햇빛 좋은 날 듣기 좋은 노래 추천해줘`
- `비 오는 날 드라이브 R&B 잔잔한`
- `운동할 때 신나는 힙합`

출력에는 항상 DB에서 나온 `track_id` 기반 트랙만 등장합니다.

## 9) 자주 터지는 문제
### A) ModuleNotFoundError: mixsense_ai_search
- **원인:** `python run_local.py`를 프로젝트 루트가 아닌 다른 위치에서 실행
- **해결:** `cd mixsense_ai_search_project` 후 실행

### B) OPENAI_API_KEY 관련 에러 / parser mode가 rule로만 뜸
- `.env`가 프로젝트 루트에 있는지 확인
- `.env`에 키가 들어있는지 확인
- 터미널을 껐다 켜고 재실행(환경변수 캐시 이슈)

### C) pandas 설치 오류
- 가상환경(.venv) 활성화 상태인지 확인 후 재설치:
```bash
pip install -r requirements.txt --no-cache-dir
```


---

## 카드 UI로 실행하기 (Cursor / Mac)

터미널에서 프로젝트 루트로 이동 후:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

브라우저가 열리면, 검색창에 질문을 입력하세요.
결과는 **썸네일 + 곡 제목 + 아티스트**만 보여주고, 카드를 클릭하면 YouTube로 이동합니다.

### (선택) '이 노래 뭐야?' 같은 식별 질의까지 지원하려면
카탈로그에 없는 곡을 “설명/구절”로 찾는 기능은 YouTube 검색이 필요합니다.

- `OPENAI_API_KEY`는 Intent Parser용 (필수)
- `YOUTUBE_API_KEY`는 YouTube Data API v3 검색용 (선택)

`.env`에 아래를 추가:

```env
YOUTUBE_API_KEY="YOUR_GOOGLE_API_KEY"
```

YouTube API 키가 없으면, 앱이 **YouTube 검색 페이지 링크**로 대신 안내합니다.


### UX 기본값
- 추천/탐색: 기본 5개
- 식별("이 노래 뭐야?"): TOP3
