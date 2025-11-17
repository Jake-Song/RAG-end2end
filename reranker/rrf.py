f"""
Reciprocal Rank Fusion (RRF) 구현
RRF(d) = sum(1 / (k + rank_i(d)))
d: 문서
k: 상수
rank_i(d): 리트리버 i에 대한 문서 d의 순위
"""
from langchain.schema import Document


class ReciprocalRankFusion:
    
    k = 60
       
    @classmethod
    def _calculate_score(cls, rank: int) -> float:
        score = 1 / (cls.k + rank) 
        return score
    
    @classmethod
    def calculate_rank_score(cls, docs: list[Document]) -> list[Document]:
        for idx, doc in enumerate(docs):
            rank = idx + 1
            doc.metadata['rank_score'] = cls._calculate_score(rank)
        return docs

    @classmethod
    def get_rrf_docs(cls, docs: list[Document], cutoff: int) -> list[Document]:

        merged = {}
        for doc in docs:
            current_id = doc.metadata['id']

            new_id = True if current_id not in merged.keys() else False
            bucket = merged.setdefault(current_id, doc.model_copy(deep=True))
            
            if not new_id:
                bucket.metadata['rank_score'] += doc.metadata['rank_score']

            else:
                bucket.metadata['rank_score'] = doc.metadata['rank_score']

        merged_docs = list(merged.values())

        merged_docs_sorted = sorted(merged_docs, key=lambda x: x.metadata['rank_score'], reverse=True)

        rrf_docs = merged_docs_sorted[:cutoff]
        return rrf_docs    
    
  
