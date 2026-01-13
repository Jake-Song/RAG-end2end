import pandas as pd
from tqdm import tqdm
from rag_basic.rag import rag_bot_batch
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
project_root = Path(__file__).parent.parent

def sample_data(df: pd.DataFrame, sample_size: int = 100) -> pd.DataFrame:
    return df.sample(n=sample_size)

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
    df_eval = pd.read_csv(project_root / "outputs" / "SPRI_ALL_synthetic_single_chunk.csv")
    df_eval = df_eval[df_eval['query'].notna()]

    df_sample = sample_data(df_eval, sample_size=10)
    outputs = generate_outputs(df_sample)
    df_sample = add_outputs_to_df(df_sample, outputs)
    
    df_sample.to_csv(project_root / "outputs" / "SPRI_ALL_eval.csv", index=False)
