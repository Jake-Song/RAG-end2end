import numpy as np
from openai import OpenAI

# -----------------------------------------------------
# 1. 환경 설정 및 API 클라이언트 초기화
# -----------------------------------------------------
# API 키가 환경 변수 OPENAI_API_KEY에 설정되어 있어야 합니다.
try:
    client = OpenAI()
except Exception as e:
    print(f"OpenAI 클라이언트 초기화 오류: {e}")
    print("API 키가 환경 변수에 올바르게 설정되었는지 확인해주세요.")
    exit()

# 최신 임베딩 모델 사용 (OpenAI 권장)
EMBEDDING_MODEL = "text-embedding-3-small"


# -----------------------------------------------------
# 2. 텍스트를 OpenAI 임베딩 벡터로 변환하는 함수
# -----------------------------------------------------
def get_openai_embedding(text: str) -> np.ndarray:
    """
    OpenAI API를 사용하여 주어진 텍스트의 임베딩 벡터를 반환합니다.
    """
    # 텍스트가 너무 길면 API 호출 시 오류가 발생할 수 있으므로 앞부분만 사용
    response = client.embeddings.create(
        input=[text.replace("\n", " ")[:8192]],
        model=EMBEDDING_MODEL
    )
    # API 응답에서 임베딩 벡터만 추출하여 NumPy 배열로 반환
    return np.array(response.data[0].embedding)


# -----------------------------------------------------
# 3. 코사인 유사도를 계산하는 함수 (NumPy 사용)
# -----------------------------------------------------
def calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    두 벡터(NumPy 배열) 간의 코사인 유사도를 계산하여 반환합니다.
    """
    # 벡터 내적 (Dot Product)
    dot_product = np.dot(vec_a, vec_b)
    
    # 벡터 노름(Norm) 또는 크기
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    # 코사인 유사도 계산: (A · B) / (||A|| * ||B||)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


# -----------------------------------------------------
# 4. 실행 예시
# -----------------------------------------------------
if __name__ == "__main__":
    # 유사도 평가 대상 텍스트
    text_answer_a = "우리는 어제 저녁 식사로 매콤한 불고기를 먹었다."
    text_answer_b = "어젯밤 우리는 저녁으로 한국식 바비큐인 불고기를 즐겼다."
    text_unrelated_c = "프랑스의 수도는 파리이며 에펠탑이 유명한 관광지이다."
    
    print(f"--- 텍스트 임베딩 생성 시작 ({EMBEDDING_MODEL}) ---")
    
    # 텍스트 임베딩 벡터 생성
    vector_a = get_openai_embedding(text_answer_a)
    vector_b = get_openai_embedding(text_answer_b)
    vector_c = get_openai_embedding(text_unrelated_c)
    
    print("--- 임베딩 생성 완료 ---")
    print(f"임베딩 벡터 차원: {vector_a.shape[0]}차원")
    print("-" * 40)

    # 유사한 답변 A와 B의 유사도 계산
    similarity_ab = calculate_cosine_similarity(vector_a, vector_b)
    
    # 관련 없는 답변 A와 C의 유사도 계산
    similarity_ac = calculate_cosine_similarity(vector_a, vector_c)
    
    # 결과 출력
    print(f"답변 A: \"{text_answer_a}\"")
    print(f"답변 B: \"{text_answer_b}\"")
    print(f"답변 C: \"{text_unrelated_c}\"")
    print("-" * 40)
    print(f"✅ [유사한 답변] A와 B의 코사인 유사도: {similarity_ab:.4f}")
    print(f"❌ [관련 없는 답변] A와 C의 코사인 유사도: {similarity_ac:.4f}")