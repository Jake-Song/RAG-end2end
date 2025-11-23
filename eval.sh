#! /bin/bash

# -----------------------------------------------------
# 데이터 답변 생성(Bot Answers)
# -----------------------------------------------------
echo "데이터 답변 생성(Bot Answers) 시작"
uv run python -m scripts.bot_answer 
echo "데이터 답변 생성(Bot Answers) 완료"

# -----------------------------------------------------
# LLM as Judge(correctness) 평가
# -----------------------------------------------------
echo "LLM as Judge 평가 시작"
uv run python -m scripts.correct_eval 
echo "LLM as Judge 평가 완료"

# -----------------------------------------------------
# summary_eval(Recall, F1 Score)
# -----------------------------------------------------
echo "summary_eval(Recall, F1 Score) 시작"
uv run python -m scripts.summary_eval 
echo "summary_eval(Recall, F1 Score) 완료"