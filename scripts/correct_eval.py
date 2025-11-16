import pandas as pd
from evaluators.llm_evaluator import CorrectnessEvaluator
from config import output_path_prefix
from rag import rag_bot_batch

correctness_evaluator = CorrectnessEvaluator()
df = pd.read_csv(f"{output_path_prefix}_synthetic.csv")

def generate_outputs(df: pd.DataFrame) -> list[dict]:
    inputs = []
   
    for index, row in df.iterrows():
        inputs.append(row["query"])

    outputs = rag_bot_batch(inputs)

    return outputs
    
def add_outputs_to_df(df: pd.DataFrame, outputs: list[dict]) -> pd.DataFrame:
    if len(df) != len(outputs):
        raise ValueError("The number of rows in the dataframe and the number of outputs do not match")
    
    df_synthetic = df.copy()
    df_synthetic['outputs.answer'] = [output["answer"] for output in outputs]
    df_synthetic['outputs.page_number'] = [output["page_number"] for output in outputs]

    return df_synthetic
        
def main():
    df_test = df.iloc[3:5].copy()
    outputs = generate_outputs(df_test)
    df_synthetic = add_outputs_to_df(df_test, outputs)

    for i, row in df_synthetic.iterrows():
        inputs = {"query": row["query"]}
        outputs = {"answer": row["outputs.answer"]}
        reference_outputs = {"answer": row["answer"]}
        correctness = correctness_evaluator.correctness(inputs, outputs, reference_outputs)
        df_synthetic.loc[i, 'correctness'] = correctness["correctness"]
        df_synthetic.loc[i, 'explanation'] = correctness["explanation"]

    df_synthetic.to_csv(f"{output_path_prefix}_eval.csv", index=False)


if __name__ == "__main__":
    main()

