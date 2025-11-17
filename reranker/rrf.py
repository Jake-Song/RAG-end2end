class RRF:
    
    k = 60
       
    @classmethod
    def _calculate_score(cls, rank):
        score = 1 / (cls.k + rank) 
        return score
    
    @classmethod
    def get_rrf_docs(cls, one, other, cutoff):

        for idx, (one_doc, other_doc) in enumerate(zip(one, other)):
            rank = idx + 1
            one_doc.metadata['rank_score'] = cls._calculate_score(rank)
            other_doc.metadata['rank_score'] = cls._calculate_score(rank)   

        docs = one + other

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
    
  
