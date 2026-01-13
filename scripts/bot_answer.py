import pandas as pd
from tqdm import tqdm
from config import output_path_prefix
from rag_basic.rag import rag_bot_batch

def generate_outputs(df: pd.DataFrame, batch_size: int = 10) -> list[dict]:
    queries = df["query"].tolist()
    outputs = []
   
    for i in tqdm(range(0, len(queries), batch_size), desc="Generating answers"):
        batch = queries[i:i + batch_size]
        batch_outputs = rag_bot_batch(batch)
        outputs.extend(batch_outputs)

    return outputs
    
def add_outputs_to_df(df: pd.DataFrame, outputs: list[dict]) -> pd.DataFrame:
    if len(df) != len(outputs):
        raise ValueError("The number of rows in the dataframe and the number of outputs do not match")
    
    df_eval = df.copy()
    df_eval['outputs_answer'] = [output["answer"] for output in outputs]
    df_eval['outputs_chunk_ids'] = [output["chunk_ids"] for output in outputs]
    df_eval['retrieved_contexts'] = [output["context"] for output in outputs]

    return df_eval

if __name__ == "__main__":
    df_eval = pd.read_csv(f"{output_path_prefix}_synthetic_single_chunk.csv")
    df_eval = df_eval[df_eval['query'].notna()]
    outputs = generate_outputs(df_eval)
    df_eval = add_outputs_to_df(df_eval, outputs)
    df_eval.to_csv(f"{output_path_prefix}_eval.csv", index=False)
