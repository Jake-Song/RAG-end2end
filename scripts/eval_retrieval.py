"""
Recall, F1 Score 평가
검색된(retrieved) 문서 데이터와 정답(answer) 문서 데이터를 비교하여 Retrieval Metric(Recall, F1 Score)을 계산
Note: 검색된 문서 수와 정답 문서 데이터 수가 같은 경우 Recall과 Precision이 같기 때문에 F1 Score의 의미가 없음.
"""

import pandas as pd
from tqdm import tqdm
from config import output_path_prefix

def recall(df: pd.DataFrame) -> dict:
    true_positives = 0
    false_negatives = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Calculating recall"):
        # 중복 페이지 제거
        # reference_page_number = list({int(page) for page in row["page_number"].strip("[]").split(",")})
        reference_page_number = row["page_number"]
        retrieved_page_number = list({int(page) for page in row["outputs.page_number"].strip("[]").split(",")})
       
        if reference_page_number in retrieved_page_number:
            true_positives += 1
        else:
            print("index: ", i)
            print(row["query"])
            print(row["page_number"])
            print(row["outputs.page_number"])
            false_negatives += 1

    print(f"True Positives: {true_positives}, False Negatives: {false_negatives}")

    recall = true_positives / (true_positives + false_negatives)
    return {"recall": recall}

def f1_score(df: pd.DataFrame) -> dict:
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Calculating F1 score"):
        # 중복 페이지 제거
        # reference_page_number = list({int(page) for page in row["page_number"].strip("[]").split(",")})
        reference_page_number = row["page_number"]
        retrieved_page_number = list({int(page) for page in row["outputs.page_number"].strip("[]").split(",")})
        # print("reference_page_number : ", reference_page_number)
        # print("retrieved_page_number : ", retrieved_page_number)
        # print("-"*100)
        # 정답지 (reference_page_numer)를 찾은 경우 true_positives +1
        # 정답지 (reference_page_numer)를 못 찾은 경우 false_negatives +1, false_positives +1
        # 정답지 (reference_page_numer)를 찾았지만 오답을 찾은 경우 false_positives +1

        # precision
        for page in retrieved_page_number:
            if page == reference_page_number:
                true_positives += 1
            else:
                false_positives += 1
        
        # recall
        if reference_page_number not in retrieved_page_number:
            false_negatives += 1
                
    if true_positives == 0:
        return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0}

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
        
    f1_score = 2 * (precision * recall) / (precision + recall)

    return {"f1_score": f1_score, "precision": precision, "recall": recall}

def main():
    # df = pd.read_csv(f"{output_path_prefix}_eval.csv")
    # df_correct = pd.read_csv(f"{output_path_prefix}_eval_correct.csv")
    df = pd.read_csv(f"{output_path_prefix}_eval_correct_adaptive_20260110_234259.csv")
    recall_result = recall(df)
    print(f"Recall: {recall_result['recall']}")
    f1_score_result = f1_score(df)
    print(f"F1 Score: {f1_score_result['f1_score']}")
    print(f"Precision: {f1_score_result['precision']}")
    print(f"Recall: {f1_score_result['recall']}")
    
if __name__ == "__main__":
    main()