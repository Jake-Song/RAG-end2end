import pandas as pd
from config import output_path_prefix

df = pd.read_csv(f"{output_path_prefix}_eval.csv")

def recall(df: pd.DataFrame) -> dict:
    true_positives = 0
    false_negatives = 0

    for _, row in df.iterrows():
        if int(row["page_number"]) not in [int(page) for page in row["outputs.page_number"].strip("[]").split(",")]:
            false_negatives += 1
        else: 
            true_positives += 1

    recall = true_positives / (true_positives + false_negatives)
    return {"recall": recall}

def f1_score(df: pd.DataFrame) -> dict:
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for _, row in df.iterrows():
        if row["correctness"] == True:            
            true_positives += 1
        elif row["correctness"] == False:
            false_positives += 1

    precision = true_positives / (true_positives + false_positives)

    if true_positives == 0:
        return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0}

    true_positives = 0
    false_positives = 0

    for _, row in df.iterrows():
        
        if int(row["page_number"]) not in [int(page) for page in row["outputs.page_number"].strip("[]").split(",")]:
            false_negatives += 1
        else: 
            true_positives += 1

    recall = true_positives / (true_positives + false_negatives)
        
    f1_score = 2 * (precision * recall) / (precision + recall)

    return {"f1_score": f1_score}

def main():
    recall_result = recall(df)
    print(f"Recall: {recall_result['recall']}")
   
    f1_score_result = f1_score(df)
    print(f"F1 Score: {f1_score_result['f1_score']}")
   
if __name__ == "__main__":
    main()