"""
langsmith 데이터 평가
"""

import pandas as pd
from langsmith import Client
from rag import rag_bot_invoke
from evaluators.f1_score import f1_score_summary_evaluator
from config import output_path_prefix, FILE_NAME
from evaluators.llm_evaluator import CorrectnessEvaluator

correctness_evaluator = CorrectnessEvaluator()

correct = correctness_evaluator.correctness

client = Client()

def create_dataset(df: pd.DataFrame, dataset_name: str) -> None:

    input_keys = ['query'] # replace with your input column names
    output_keys = ['answer', 'page_number'] # replace with your output column names
   
    _ = client.upload_dataframe(
        df=df,
        input_keys=input_keys,
        output_keys=output_keys,
        name=dataset_name,
        description="Dataset created from a parquet file",
        data_type="kv" # The default
    )

def target(inputs: dict) -> dict:
    print("inputs", inputs)
    return rag_bot_invoke(inputs["query"])

def evaluate(dataset_name: str, limit: int) -> dict:
    examples = client.list_examples(dataset_name=dataset_name, limit=limit)

    experiment_results = client.evaluate(
        target,
        data=examples,
        evaluators=[correct],
        summary_evaluators=[f1_score_summary_evaluator],
        experiment_prefix="rag-f1",
        metadata={"version": "langgraph, gpt-5-nano"},
    )

    return experiment_results

def main():
    
    # df = pd.read_csv(f"{output_path_prefix}_synthetic.csv")
    dataset_name = f"{FILE_NAME}_synthetic_dataset-gpt-5"
    # create_dataset(df, dataset_name)
    print("데이터셋 생성")
    experiment_results = evaluate(dataset_name, 1)
    print("평가 완료")
    experiment_results.to_pandas().to_csv(f"{output_path_prefix}_evaluation.csv", index=False)
    print("평가 결과 저장")
    print("✅ 모든 작업 완료")

if __name__ == "__main__":
    main()