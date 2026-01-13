"""
Local LLM 데이터 평가
LLM as Judge 방식
1. correctness 평가
"""

import pandas as pd
from tqdm import tqdm
from evaluators.llm_evaluator import CorrectnessEvaluator
from pathlib import Path
project_root = Path(__file__).parent.parent
import time
from dotenv import load_dotenv
load_dotenv()

correctness_evaluator = CorrectnessEvaluator()

def main():
    # df_eval = pd.read_csv(f"{output_path_prefix}_eval.csv")
    df_eval = pd.read_csv(project_root / "outputs" / "SPRI_ALL_eval.csv")
    query_list = df_eval["query"].to_list()
    outputs_answer_list = df_eval["outputs_answer"].to_list()
    answer_list = df_eval["answer"].to_list()
    
    start_time = time.time()
    print(f"배치 평가 시작: {start_time}")
    
    batch_size = 10
    results = []
    for i in tqdm(range(0, len(query_list), batch_size), desc="Evaluating correctness"):
        batch_queries = query_list[i:i + batch_size]
        batch_outputs = outputs_answer_list[i:i + batch_size]
        batch_answers = answer_list[i:i + batch_size]
        
        batch_results = correctness_evaluator.correctness_batch(batch_queries, batch_outputs, batch_answers)
        results.extend(batch_results)

    end_time = time.time()
    print(f"배치 평가 완료: {end_time - start_time}초")
    df_eval['correctness'] = [result["correctness"] for result in results]
    df_eval['explanation'] = [result["explanation"] for result in results]

    # df_eval.to_csv(f"{output_path_prefix}_eval_correct.csv", index=False)
    df_eval.to_csv(project_root / "outputs" / "SPRI_ALL_eval_correct.csv", index=False)
    print("평가 결과 저장 완료")
if __name__ == "__main__":
    main()

