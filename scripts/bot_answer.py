import pandas as pd
from config import output_path_prefix
from rag_basic.rag import rag_bot_batch

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
    df_eval['retrieved_contexts'] = [output["context"] for output in outputs]

    return df_eval

if __name__ == "__main__":
    df_eval = pd.read_csv(f"{output_path_prefix}_synthetic_single.csv")
    df_eval = df_eval[df_eval['query'].notna()]
    outputs = generate_outputs(df_eval)
    df_eval = add_outputs_to_df(df_eval, outputs)
    df_eval.to_csv(f"{output_path_prefix}_eval.csv", index=True)
