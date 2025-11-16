from utils.embed_openai import get_openai_embedding, calculate_cosine_similarity, EMBEDDING_MODEL

def f1_score_summary_evaluator(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for output_dict, reference_output_dict in zip(outputs, reference_outputs):
                
        output_answer = output_dict["answer"]
        reference_output_answer = reference_output_dict["answer"]

        output_page_number = output_dict["page_number"]
        reference_output_page_number = reference_output_dict["page_number"]

        # 문장 비교 방법 추가(코사인 유사도)
        output = get_openai_embedding(output_answer)
        reference_output = get_openai_embedding(reference_output_answer)
        similarity = calculate_cosine_similarity(output, reference_output)

        if similarity > 0.7:
            true_positives += 1

        elif output_answer != reference_output_answer:
            false_positives += 1
        elif isinstance(output_page_number, (list, tuple, set)):
                if reference_output_page_number in output_page_number:
                    true_positives += 1
                else:
                    false_negatives += 1
        
    if true_positives == 0:
        return {"key": "f1_score", "score": 0.0}

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    f1_score = 2 * (precision * recall) / (precision + recall)

    return {"results": [{"key": "f1_score", "score": f1_score}, {"key": "precision", "score": precision}, {"key": "recall", "score": recall}]}
   

