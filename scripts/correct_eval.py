"""
Local LLM 데이터 평가
LLM as Judge 방식
1. correctness 평가
"""

import pandas as pd
from evaluators.llm_evaluator import CorrectnessEvaluator
from config import output_path_prefix

correctness_evaluator = CorrectnessEvaluator()

def main():
    df_eval = pd.read_csv(f"{output_path_prefix}_eval.csv")
    
    for i, row in df_eval.iterrows():
        inputs = {"query": row["query"]}
        outputs = {"answer": row["outputs.answer"]}
        reference_outputs = {"answer": row["answer"]}
        correctness = correctness_evaluator.correctness(inputs, outputs, reference_outputs)
        df_eval.loc[i, 'correctness'] = correctness["correctness"]
        df_eval.loc[i, 'explanation'] = correctness["explanation"]

    df_eval.to_csv(f"{output_path_prefix}_eval_correct.csv", index=True)

if __name__ == "__main__":
    main()

