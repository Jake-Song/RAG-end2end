"""
Recall, F1 Score 평가
검색된(retrieved) 문서 데이터와 정답(answer) 문서 데이터를 비교하여 Retrieval Metric(Recall, F1 Score)을 계산
Note: 검색된 문서 수와 정답 문서 데이터 수가 같은 경우 Recall과 Precision이 같기 때문에 F1 Score의 의미가 없음.
"""

import pandas as pd
from config import output_path_prefix

def recall(df: pd.DataFrame) -> dict:
    true_positives = 0
    false_negatives = 0

    for _, row in df.iterrows():
        # 중복 페이지 제거
        reference_page_number = list({int(page) for page in row["page_number"].strip("[]").split(",")})
        retrieved_page_number = list({int(page) for page in row["outputs.page_number"].strip("[]").split(",")})
        for page in reference_page_number:
            if page in retrieved_page_number:
                true_positives += 1
            else:
                false_negatives += 1

    print(f"True Positives: {true_positives}, False Negatives: {false_negatives}")

    recall = true_positives / (true_positives + false_negatives)
    return {"recall": recall}

def f1_score(df: pd.DataFrame) -> dict:
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for _, row in df.iterrows():
        # 중복 페이지 제거
        reference_page_number = list({int(page) for page in row["page_number"].strip("[]").split(",")})
        retrieved_page_number = list({int(page) for page in row["outputs.page_number"].strip("[]").split(",")})
        
        # 정답지 (reference_page_numer)를 찾은 경우 true_positives +1
        # 정답지 (reference_page_numer)를 못 찾은 경우 false_negatives +1, false_positives +1
        # 정답지 (reference_page_numer)를 찾았지만 오답을 찾은 경우 false_positives +1
        for page in retrieved_page_number:
            if page in reference_page_number:
                true_positives += 1
            else:
                false_positives += 1
        
        for page in reference_page_number:
            if page not in retrieved_page_number:
                false_negatives += 1
                
    if true_positives == 0:
        return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0}

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
        
    f1_score = 2 * (precision * recall) / (precision + recall)

    return {"f1_score": f1_score, "precision": precision, "recall": recall}

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
    df = pd.read_csv(f"{output_path_prefix}_eval.csv")
    df_correct = pd.read_csv(f"{output_path_prefix}_eval_correct.csv")
    recall_result = recall(df)
    print(f"Recall: {recall_result['recall']}")
    f1_score_result = f1_score(df)
    print(f"F1 Score: {f1_score_result['f1_score']}")
    print(f"Precision: {f1_score_result['precision']}")
    print(f"Recall: {f1_score_result['recall']}")
    correctness_result = correctness(df_correct)
    print(f"Correctness: {correctness_result['correctness']}")
if __name__ == "__main__":
    main()