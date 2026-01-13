import pandas as pd
from pathlib import Path
project_root = Path(__file__).parent.parent

def correctness(df: pd.DataFrame) -> dict:
    correctness_true = 0
    correctness_false = 0
    for _, row in df.iterrows():
        if row["correctness"] == True:
            correctness_true += 1
        else:
            correctness_false += 1
    correctness = correctness_true / (correctness_true + correctness_false)
    return {"correctness": correctness}    

def main():
    df_correct = pd.read_csv(project_root / "outputs" / "SPRI_ALL_eval_correct.csv")
   
    correctness_result = correctness(df_correct)
    print(f"Correctness: {correctness_result['correctness']}")
if __name__ == "__main__":
    main()