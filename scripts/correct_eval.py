"""
Local LLM 데이터 평가
LLM as Judge 방식
1. correctness 평가
"""

import pandas as pd
from evaluators.llm_evaluator import CorrectnessEvaluator
from config import output_path_prefix
import time

correctness_evaluator = CorrectnessEvaluator()

def main():
    df_eval = pd.read_csv(f"{output_path_prefix}_eval.csv")
    query_list = df_eval["query"].to_list()
    outputs_answer_list = df_eval["outputs.answer"].to_list()
    answer_list = df_eval["answer"].to_list()
    
    start_time = time.time()
    print(f"배치 평가 시작: {start_time}")
    results = correctness_evaluator.correctness_batch(query_list, outputs_answer_list, answer_list)
    end_time = time.time()
    print(f"배치 평가 완료: {end_time - start_time}초")
    df_eval['correctness'] = [result["correctness"] for result in results]
    df_eval['explanation'] = [result["explanation"] for result in results]

    df_eval.to_csv(f"{output_path_prefix}_eval_correct.csv", index=True)
    print("평가 결과 저장 완료")
if __name__ == "__main__":
    main()

