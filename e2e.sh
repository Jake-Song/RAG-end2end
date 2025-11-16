#! /bin/bash

echo "데이터 가져오기 시작"
uv run python -m scripts.fetch 
echo "데이터 가져오기 완료"

echo "데이터 파싱 시작"
uv run python -m scripts.parse 
echo "데이터 파싱 완료"

echo "데이터 청킹 시작"
uv run python -m scripts.chunk 
echo "데이터 청크링 완료"

echo "합성 데이터 생성 시작"
uv run python -m scripts.synthetic_data 
echo "합성 데이터 생성 완료"

echo "데이터 리트리버 시작"
uv run python -m scripts.retrieve 
echo "데이터 리트리버 완료"

echo "데이터 평가(LLM as Judge) 시작"
uv run python -m scripts.correct_eval 
echo "데이터 평가(LLM as Judge) 완료"

echo "summary_eval(Recall, F1 Score) 시작"
uv run python -m scripts.summary_eval 
echo "summary_eval(Recall, F1 Score) 완료"

echo "✅ 모든 작업 완료"