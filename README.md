# RAG end to end 

## 주요 기능

- **이미지 요약**: 이미지를 추출하고 요약 정보를 생성합니다.
- **텍스트 요약**: 텍스트를 페이지별로 요약합니다.
- **하이브리드 검색**: BM25(키워드 기반) + FAISS(의미 기반) 
- **RAG workflow**: 


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
### 0. PDF 파일 다운로드
- data 폴더에 저장

### 1. config 파일 설정
```python
FILE_NAME ="file_name"  # 확장자(.pdf) 없이 기입
```

### 2. bash 파일 실행 설정
```bash
chmod +x prep.sh 
./prep.sh # 데이터 준비 청킹까지
```

### 3. Vector DB 저장
/db 폴더 노트북 참고 (FAISS, PGVector, ElasticSearch)
로컬 실행 (Docker)
```bash
# local 실행
# PGVector
cd pgvector-start-local
docker compose up -d

# ElasticSearch
curl -fsSL https://elastic.co/start-local | sh
```
### 4. Chat UI
```bash
# backend
uvicorn chat.agent:app --reload # rag agent only 
uvicorn chat.feedback:app --reload # rag agent with human feedback

# frontend
streamlit run chat/frontend.py
```


