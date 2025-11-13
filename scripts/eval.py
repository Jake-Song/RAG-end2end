import pandas as pd
from langsmith import Client
from rag import rag_bot_invoke
from evaluators.f1_score import f1_score_summary_evaluator
from dotenv import load_dotenv
load_dotenv()

# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install -qU langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("Eval")

client = Client()

def create_dataset(df: pd.DataFrame, dataset_name: str) -> None:

    input_keys = ['query'] # replace with your input column names
    output_keys = ['answers', 'page_number'] # replace with your output column names
   
    _ = client.upload_dataframe(
        df=df,
        input_keys=input_keys,
        output_keys=output_keys,
        name=dataset_name,
        description="Dataset created from a parquet file",
        data_type="kv" # The default
    )

def target(inputs: dict) -> dict:
    return rag_bot_invoke(inputs["query"])

def evaluate(dataset_name: str, limit: int) -> dict:
    examples = client.list_examples(dataset_name=dataset_name, limit=limit)

    experiment_results = client.evaluate(
        target,
        data=examples,
        # evaluators=[correct],
        summary_evaluators=[f1_score_summary_evaluator],
        experiment_prefix="rag-f1",
        metadata={"version": "langgraph, gpt-5-nano"},
    )

    return experiment_results

def main():
    
    df = pd.read_csv("outputs/synthetic_output.csv")
    dataset_name = "synthetic_dataset"
    create_dataset(df, dataset_name)
    print("Created dataset")
    experiment_results = evaluate(dataset_name, 10)
    print("Evaluated")
    experiment_results.to_pandas().to_csv(f"outputs/{dataset_name}_evaluation.csv", index=False)
    print("Saved evaluation results")

if __name__ == "__main__":
    main()