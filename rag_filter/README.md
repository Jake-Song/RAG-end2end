# Vector Search Filter

## 주요 기능
- **Agent**
- **Serach Filter For metadata**
- **FAISS 검색**
- **PGVector 검색**
- **ElasticSearch 검색**

## 워크플로우
- **Rag Filter Feedback**

![](rag_filter.png)

## Rational
- 문서가 많거나 다양한 카테고리인 경우 필터가 필요함.
- FAISS 메타데이터 검색: 단순 키워드 일치
- PGVector 메다데이터 검색: 보다 복잡한 쿼리 가능. 
  - 하지만 BM25 + Dense Vector Search 이기 때문에 각각 같은 쿼리를 적용할 필요가 생김 
- SQL 쿼리로 같은 쿼리 적용
- ElasticSearch: PGVector DB보다 간단함. SQL 쿼리 문법으로 하지 않아도 됨.
 