# RAG end to end 

## 주요 기능

- **이미지 요약**: 이미지를 추출하고 요약 정보를 생성합니다.
- **텍스트 요약**: 텍스트를 페이지별로 요약합니다.
- **하이브리드 검색**: BM25(키워드 기반) + FAISS(의미 기반) 
- **대화형 CLI**: 재로딩 없이 여러 질문을 연속으로 처리하는 대화형 인터페이스


## 아키텍처

```
┌─────────────┐
│    질문     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│   문서 검색             │
│   (Kiwi-BM25 + FAISS)   │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│   LLM 답변 생성         │
│                         │
└──────┬──────────────────┘
       │
       ▼
┌─────────────┐
│    답변     │
└─────────────┘

```

## 설치

### uv 설치

macOS/Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 패키지 설치

```bash
# 기본 패키지 설치
uv sync

# dev 패키지 포함 설치
uv sync --dev
```

환경 변수 파일:
```bash
cp .env.example .env
# env 설정
UPSTAGE_API_KEY=           # 문서 파서, 임베딩 모델, 추론 모델
OPENAI_API_KEY=            # 임베딩 모델, 추론 모델
LANGSMITH_API_KEY=         # 실행 추적

FILE_NAME=                 # 확장자 없이 파일 이름 (파일은 data 폴더 내에 위치해야 합니다.)
```

## 사용법

### 1. 대화형 CLI 모드

```bash
python rag.py
```

실행 예시:
```
Loading RAG system...
RAG system loaded in 3.45 seconds
============================================================
RAG Bot Interactive Mode
Type 'exit' or 'quit' to end the session
============================================================

Question: 주요 경제 과제는 무엇인가요?
LLM answer generation time: 1.23 seconds

Answer: 주요 경제 과제는...

Page numbers: [1, 5, 12]
------------------------------------------------------------

Question: 무역 정책에 대해 더 알려주세요
LLM answer generation time: 1.15 seconds

Answer: 무역 정책은...

Page numbers: [3, 7]
------------------------------------------------------------

Question: exit
Goodbye!
```
### 2. 프로그래밍 방식

```python
from rag import rag_bot_invoke

# 첫 호출 시 시스템 로드
result = rag_bot_invoke("경제 전망은 어떻습니까?")
print(result['answer'])
print(result['page_number'])

# 이후 호출은 로드된 리소스 재사용
result2 = rag_bot_invoke("인플레이션은 어떻습니까?")
```

### 3. 스크립트 파일 실행
```bash
# 데이터 가져오기
uv run python -m scripts.fetch

# 데이터 파싱
uv run python -m scripts.parse

# 데이터 분할
uv run python -m scripts.chunk

# 리트리버 생성
uv run python -m scripts.retrieve

# 데이터 생성 
uv run python -m scripts.synthetic_data

# RAG 평가
# 데이터셋 응답
uv run python -m scripts.bot_answer
# LLM as Judge(correctness)
uv run python -m scripts.correct_eval
# Recall, F1 score
uv run python -m scripts.summary_eval

```

### 4. bash 파일 실행
```bash
chmod +x e2e.sh
chmod +x ready_for_RAG.sh
chmod +x gen_data.sh
chmod +x eval.sh

# 파싱부터 RAG 평가까지
./e2e.sh

# 데이터 파싱부터 RAG 시스템 구축까지
./ready_for_RAG.sh

# 데이터 생성
./gen_data.sh

# RAG 평가
./eval.sh

```


