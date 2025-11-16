#! /bin/bash

# 데이터 파싱부터 평가까지 한 번에 실행합니다.
# 환경 변수 가져오기
# if [ -f .env ]; then
#     export $(grep -v '^#' .env | xargs)
# fi



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

echo "데이터 평가 시작"
uv run python -m scripts.eval 
echo "데이터 평가 완료"
echo "✅ 모든 작업 완료"