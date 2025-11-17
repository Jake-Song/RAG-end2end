"""
Test file for reranker/rrf.py

Tests the Reciprocal Rank Fusion (RRF) implementation.
"""

import pytest
from langchain_core.documents import Document
from reranker.rrf import RRF


class TestRRF:
    """Test suite for RRF class"""

    def test_calculate_score(self):
        """Test the _calculate_score method"""
        # Test with default k=60
        assert RRF._calculate_score(1) == 1 / (60 + 1)  # 1/61
        assert RRF._calculate_score(2) == 1 / (60 + 2)  # 1/62
        assert RRF._calculate_score(10) == 1 / (60 + 10)  # 1/70
        
        # Verify score decreases as rank increases
        score_1 = RRF._calculate_score(1)
        score_2 = RRF._calculate_score(2)
        assert score_1 > score_2

    def test_get_rrf_docs_basic(self):
        """Test basic RRF functionality with non-overlapping documents"""
        # Create test documents
        doc1 = Document(
            page_content="Content 1",
            metadata={"id": 1, "page": 1}
        )
        doc2 = Document(
            page_content="Content 2",
            metadata={"id": 2, "page": 1}
        )
        doc3 = Document(
            page_content="Content 3",
            metadata={"id": 3, "page": 2}
        )
        doc4 = Document(
            page_content="Content 4",
            metadata={"id": 4, "page": 2}
        )

        one = [doc1, doc2]  # First retrieval result
        other = [doc3, doc4]  # Second retrieval result

        result = RRF.get_rrf_docs(one, other, cutoff=4)

        # Should return all 4 documents
        assert len(result) == 4
        
        # Check that rank_score was added
        for doc in result:
            assert "rank_score" in doc.metadata
            assert doc.metadata["rank_score"] > 0

        # Verify documents are sorted by rank_score (descending)
        scores = [doc.metadata["rank_score"] for doc in result]
        assert scores == sorted(scores, reverse=True)

    def test_get_rrf_docs_sorting(self):
        """Test that documents are correctly sorted by rank_score"""
        # Create documents that will have different scores after merging
        doc1 = Document(page_content="Content 1", metadata={"id": 1, "page": 1})
        doc2 = Document(page_content="Content 2", metadata={"id": 2, "page": 1})
        doc1_dup = Document(page_content="Content 1 dup", metadata={"id": 1, "page": 1})
        doc3 = Document(page_content="Content 3", metadata={"id": 3, "page": 1})

        # doc1 appears in both lists (will have higher combined score)
        one = [doc1, doc2]
        other = [doc1_dup, doc3]

        result = RRF.get_rrf_docs(one, other, cutoff=10)

        # doc1 should be first (highest combined score)
        assert result[0].metadata["id"] == 1
        
        # Verify descending order
        scores = [doc.metadata["rank_score"] for doc in result]
        assert scores == sorted(scores, reverse=True)

    def test_get_rrf_docs_rank_score_calculation(self):
        """Test that rank_score is calculated correctly for each position"""
        # Create documents to test specific rank calculations
        one = [
            Document(page_content=f"Content {i}", metadata={"id": i, "page": 1})
            for i in range(3)
        ]
        other = [
            Document(page_content=f"Content {i+3}", metadata={"id": i + 3, "page": 1})
            for i in range(3)
        ]

        result = RRF.get_rrf_docs(one, other, cutoff=10)

        # Check scores for documents from 'one' list
        for i, doc in enumerate(one):
            doc_result = next(d for d in result if d.metadata["id"] == doc.metadata["id"])
            expected_score = 1 / (60 + i + 1)  # rank is i+1
            assert abs(doc_result.metadata["rank_score"] - expected_score) < 1e-10

        # Check scores for documents from 'other' list
        for i, doc in enumerate(other):
            doc_result = next(d for d in result if d.metadata["id"] == doc.metadata["id"])
            expected_score = 1 / (60 + i + 1)  # rank is i+1
            assert abs(doc_result.metadata["rank_score"] - expected_score) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

