"""
Test file for reranker/rrf.py

Tests the Reciprocal Rank Fusion (RRF) implementation.
"""

import pytest
from langchain.schema import Document
from reranker.rrf import ReciprocalRankFusion


class TestReciprocalRankFusion:
    """Test suite for ReciprocalRankFusion class"""

    def test_calculate_score(self):
        """Test the _calculate_score method"""
        # Test with default k=60
        assert ReciprocalRankFusion._calculate_score(1) == 1 / (60 + 1)  # 1/61
        assert ReciprocalRankFusion._calculate_score(2) == 1 / (60 + 2)  # 1/62
        assert ReciprocalRankFusion._calculate_score(10) == 1 / (60 + 10)  # 1/70
        
        # Verify score decreases as rank increases
        score_1 = ReciprocalRankFusion._calculate_score(1)
        score_2 = ReciprocalRankFusion._calculate_score(2)
        assert score_1 > score_2

    def test_calculate_rank_score(self):
        """Test the calculate_rank_score method"""
        # Create test documents
        docs = [
            Document(page_content=f"Content {i}", metadata={"id": i, "page": 1})
            for i in range(3)
        ]
        
        result = ReciprocalRankFusion.calculate_rank_score(docs)
        
        # Should return the same list
        assert len(result) == 3
        assert result == docs
        
        # Check that rank_score was added to each document
        for i, doc in enumerate(result):
            assert "rank_score" in doc.metadata
            expected_score = 1 / (60 + i + 1)  # rank is i+1
            assert abs(doc.metadata["rank_score"] - expected_score) < 1e-10

    def test_get_rrf_docs_basic(self):
        """Test basic RRF functionality with non-overlapping documents"""
        # Create test documents with rank_score already assigned
        doc1 = Document(
            page_content="Content 1",
            metadata={"id": 1, "page": 1, "rank_score": 1 / (60 + 1)}
        )
        doc2 = Document(
            page_content="Content 2",
            metadata={"id": 2, "page": 1, "rank_score": 1 / (60 + 2)}
        )
        doc3 = Document(
            page_content="Content 3",
            metadata={"id": 3, "page": 2, "rank_score": 1 / (60 + 1)}
        )
        doc4 = Document(
            page_content="Content 4",
            metadata={"id": 4, "page": 2, "rank_score": 1 / (60 + 2)}
        )

        docs = [doc1, doc2, doc3, doc4]

        result = ReciprocalRankFusion.get_rrf_docs(docs, cutoff=4)

        # Should return all 4 documents
        assert len(result) == 4
        
        # Check that rank_score is preserved
        for doc in result:
            assert "rank_score" in doc.metadata
            assert doc.metadata["rank_score"] > 0

        # Verify documents are sorted by rank_score (descending)
        scores = [doc.metadata["rank_score"] for doc in result]
        assert scores == sorted(scores, reverse=True)

    def test_get_rrf_docs_with_overlap(self):
        """Test RRF with overlapping documents (same id in list)"""
        # Create documents with overlapping IDs
        doc1 = Document(
            page_content="Content 1",
            metadata={"id": 1, "page": 1, "rank_score": 1 / (60 + 1)}
        )
        doc2 = Document(
            page_content="Content 2",
            metadata={"id": 2, "page": 1, "rank_score": 1 / (60 + 2)}
        )
        doc2_dup = Document(
            page_content="Content 2 duplicate",
            metadata={"id": 2, "page": 1, "rank_score": 1 / (60 + 1)}
        )
        doc3 = Document(
            page_content="Content 3",
            metadata={"id": 3, "page": 2, "rank_score": 1 / (60 + 2)}
        )

        docs = [doc1, doc2, doc2_dup, doc3]

        result = ReciprocalRankFusion.get_rrf_docs(docs, cutoff=10)

        # Should have 3 unique documents (id 1, 2, 3)
        assert len(result) == 3
        
        # Find the merged document with id=2
        doc2_merged = next(doc for doc in result if doc.metadata["id"] == 2)
        
        # The merged document should have combined rank_score
        # doc2 had score 1/(60+2) = 1/62
        # doc2_dup had score 1/(60+1) = 1/61
        # Combined: 1/62 + 1/61
        expected_score = 1 / (60 + 2) + 1 / (60 + 1)
        assert abs(doc2_merged.metadata["rank_score"] - expected_score) < 1e-10

    def test_get_rrf_docs_cutoff(self):
        """Test that cutoff parameter limits the number of returned documents"""
        # Create 10 documents with rank_score
        docs = [
            Document(
                page_content=f"Content {i}",
                metadata={"id": i, "page": 1, "rank_score": 1 / (60 + i + 1)}
            )
            for i in range(10)
        ]

        # Test with cutoff=3
        result = ReciprocalRankFusion.get_rrf_docs(docs, cutoff=3)
        assert len(result) == 3

        # Test with cutoff=5
        result = ReciprocalRankFusion.get_rrf_docs(docs, cutoff=5)
        assert len(result) == 5

        # Test with cutoff larger than available documents
        result = ReciprocalRankFusion.get_rrf_docs(docs, cutoff=20)
        assert len(result) == 10  # Should return all available

    def test_get_rrf_docs_empty_list(self):
        """Test RRF with empty input list"""
        result = ReciprocalRankFusion.get_rrf_docs([], cutoff=10)
        assert len(result) == 0

    def test_get_rrf_docs_sorting(self):
        """Test that documents are correctly sorted by rank_score"""
        # Create documents that will have different scores after merging
        doc1 = Document(
            page_content="Content 1",
            metadata={"id": 1, "page": 1, "rank_score": 1 / (60 + 1)}
        )
        doc2 = Document(
            page_content="Content 2",
            metadata={"id": 2, "page": 1, "rank_score": 1 / (60 + 2)}
        )
        doc1_dup = Document(
            page_content="Content 1 dup",
            metadata={"id": 1, "page": 1, "rank_score": 1 / (60 + 1)}
        )
        doc3 = Document(
            page_content="Content 3",
            metadata={"id": 3, "page": 1, "rank_score": 1 / (60 + 2)}
        )

        # doc1 appears twice (will have higher combined score)
        docs = [doc1, doc2, doc1_dup, doc3]

        result = ReciprocalRankFusion.get_rrf_docs(docs, cutoff=10)

        # doc1 should be first (highest combined score: 1/61 + 1/61)
        assert result[0].metadata["id"] == 1
        
        # Verify descending order
        scores = [doc.metadata["rank_score"] for doc in result]
        assert scores == sorted(scores, reverse=True)

    def test_get_rrf_docs_preserves_metadata(self):
        """Test that original metadata is preserved in merged documents"""
        doc1 = Document(
            page_content="Content 1",
            metadata={"id": 1, "page": 1, "source": "test.pdf", "rank_score": 1 / (60 + 1)}
        )
        doc2 = Document(
            page_content="Content 2",
            metadata={"id": 2, "page": 2, "source": "test.pdf", "rank_score": 1 / (60 + 1)}
        )

        result = ReciprocalRankFusion.get_rrf_docs([doc1, doc2], cutoff=10)

        # Check that original metadata is preserved
        doc1_result = next(doc for doc in result if doc.metadata["id"] == 1)
        assert doc1_result.metadata["page"] == 1
        assert doc1_result.metadata["source"] == "test.pdf"
        assert "rank_score" in doc1_result.metadata

    def test_get_rrf_docs_multiple_overlaps(self):
        """Test RRF with multiple overlapping documents"""
        # Create documents where multiple IDs appear multiple times
        docs = []
        for i in range(5):
            docs.append(Document(
                page_content=f"Content {i}",
                metadata={"id": i, "page": 1, "rank_score": 1 / (60 + i + 1)}
            ))
        # Add duplicates for first 3
        for i in range(3):
            docs.append(Document(
                page_content=f"Content {i} dup",
                metadata={"id": i, "page": 1, "rank_score": 1 / (60 + i + 1)}
            ))

        result = ReciprocalRankFusion.get_rrf_docs(docs, cutoff=10)

        # Should have 5 unique documents (ids 0-4)
        assert len(result) == 5

        # First 3 should have combined scores
        for i in range(3):
            doc = next(doc for doc in result if doc.metadata["id"] == i)
            # Score from first occurrence + score from duplicate
            expected_score = 1 / (60 + i + 1) + 1 / (60 + i + 1)
            assert abs(doc.metadata["rank_score"] - expected_score) < 1e-10

        # Last 2 should have single scores
        for i in range(3, 5):
            doc = next(doc for doc in result if doc.metadata["id"] == i)
            expected_score = 1 / (60 + i + 1)
            assert abs(doc.metadata["rank_score"] - expected_score) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

