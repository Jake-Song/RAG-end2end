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
## 파일 구조

```
.
├── rag.py                      # 메인 RAG 시스템
├── scripts/
│   ├── retrieve.py             # 리트리버 생성/로딩
│   └── eval.py                 # 평가 스크립트
├── utils/
│   └── utils.py                # 유틸리티 함수
├── config.py                   # 설정
└── dev/rag_pipeline/           # 개발용 노트북
```

## 설치
uv 설치 
```bash
# 의존성 설치
pip install langchain langchain-openai langchain-upstage langgraph

# 또는 uv 사용 시
uv sync
uv sync --dev
```

환경 변수 설정:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export UPSTAGE_API_KEY="your-upstage-api-key"
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

### 2. 단일 질문 모드

빠른 일회성 질의:

```bash
python rag.py "한국 경제의 글로벌 리스크와 과제는 무엇인가?"
```

### 3. 프로그래밍 방식 사용

#### 다른 스크립트에서 임포트 (지연 로딩으로 빠른 임포트):

```python
from rag import rag_bot_invoke

# 첫 호출 시 시스템 로드
result = rag_bot_invoke("경제 전망은 어떻습니까?")
print(result['answer'])
print(result['page_number'])

# 이후 호출은 로드된 리소스 재사용
result2 = rag_bot_invoke("인플레이션은 어떻습니까?")
```




