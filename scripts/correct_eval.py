"""
Local LLM 데이터 평가
LLM as Judge 방식
1. correctness 평가
"""

import pandas as pd
from evaluators.llm_evaluator import CorrectnessEvaluator
from config import output_path_prefix
from rag import rag_bot_batch, rag_bot_invoke

correctness_evaluator = CorrectnessEvaluator()
df = pd.read_csv(f"{output_path_prefix}_synthetic.csv")

df = df[df['query'].notna()]

def generate_outputs(df: pd.DataFrame) -> list[dict]:
    inputs = []
   
    for _, row in df.iterrows():
        inputs.append(row["query"])

    outputs = rag_bot_batch(inputs)

    return outputs
    
def add_outputs_to_df(df: pd.DataFrame, outputs: list[dict]) -> pd.DataFrame:
    if len(df) != len(outputs):
        raise ValueError("The number of rows in the dataframe and the number of outputs do not match")
    
    df_eval = df.copy()
    df_eval['outputs.answer'] = [output["answer"] for output in outputs]
    df_eval['outputs.page_number'] = [output["page_number"] for output in outputs]

    return df_eval
        
def main():
    df_eval = df.copy()
    outputs = generate_outputs(df_eval)
    df_eval = add_outputs_to_df(df_eval, outputs)

    for i, row in df_eval.iterrows():
        inputs = {"query": row["query"]}
        outputs = {"answer": row["outputs.answer"]}
        reference_outputs = {"answer": row["answer"]}
        correctness = correctness_evaluator.correctness(inputs, outputs, reference_outputs)
        df_eval.loc[i, 'correctness'] = correctness["correctness"]
        df_eval.loc[i, 'explanation'] = correctness["explanation"]

    df_eval.to_csv(f"{output_path_prefix}_eval.csv", index=False)


if __name__ == "__main__":
    main()

