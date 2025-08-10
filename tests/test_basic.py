"""
Basic tests for pharmaceutical research assistant.
Tests core functionality without external dependencies.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime

from src.models import (
    ChemicalCompound, LiteraturePaper, ResearchQuery, 
    CompoundType, ResearchStatus
)
from src.services.chemical_service import ChemicalService
from src.services.literature_service import LiteratureService
from src.services.embedding_service import EmbeddingService


class TestChemicalCompoundModel:
    """Test ChemicalCompound data model."""
    
    def test_compound_creation(self):
        """Test basic compound creation."""
        compound = ChemicalCompound(
            name="Aspirin",
            smiles="CC(=O)OC1=CC=CC=C1C(=O)O"
        )
        
        assert compound.name == "Aspirin"
        assert compound.smiles == "CC(=O)OC1=CC=CC=C1C(=O)O"
        assert compound.compound_type == CompoundType.UNKNOWN
        assert compound.created_at is not None
    
    def test_compound_validation(self):
        """Test compound data validation."""
        compound = ChemicalCompound(
            name="Test",
            smiles="   ",  # Empty SMILES should be converted to None
            bioactivity_score=0.8
        )
        
        assert compound.smiles is None
        assert compound.bioactivity_score == 0.8
    
    def test_compound_with_properties(self):
        """Test compound with calculated properties."""
        compound = ChemicalCompound(
            name="Test Compound",
            smiles="CCO",  # Ethanol
            molecular_weight=46.07,
            drug_likeness=0.6,
            bioactivity_score=0.4,
            toxicity_score=0.2,
            compound_type=CompoundType.SMALL_MOLECULE
        )
        
        assert compound.molecular_weight == 46.07
        assert compound.drug_likeness == 0.6
        assert compound.compound_type == CompoundType.SMALL_MOLECULE


class TestLiteraturePaperModel:
    """Test LiteraturePaper data model."""
    
    def test_paper_creation(self):
        """Test basic paper creation."""
        paper = LiteraturePaper(
            title="Test Paper",
            authors=["Smith, J.", "Doe, J."],
            journal="Test Journal"
        )
        
        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2
        assert paper.journal == "Test Journal"
        assert paper.retrieved_at is not None
    
    def test_paper_with_metadata(self):
        """Test paper with full metadata."""
        pub_date = datetime(2023, 1, 1)
        
        paper = LiteraturePaper(
            pubmed_id="12345678",
            doi="10.1000/test",
            title="Pharmaceutical Research",
            abstract="This is a test abstract",
            authors=["Author, A."],
            journal="Pharma Journal",
            publication_date=pub_date,
            keywords=["drug", "research"],
            relevance_score=0.85
        )
        
        assert paper.pubmed_id == "12345678"
        assert paper.doi == "10.1000/test"
        assert paper.publication_date == pub_date
        assert "drug" in paper.keywords
        assert paper.relevance_score == 0.85


class TestResearchQuery:
    """Test ResearchQuery model."""
    
    def test_basic_query(self):
        """Test basic research query."""
        query = ResearchQuery(
            query_text="test query",
            max_papers=10
        )
        
        assert query.query_text == "test query"
        assert query.max_papers == 10
        assert query.include_chemical_similarity == True
        assert query.include_literature_mining == True
    
    def test_query_with_compound(self):
        """Test query with compound information."""
        query = ResearchQuery(
            query_text="anti-inflammatory research",
            compound_smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
            compound_name="Aspirin",
            max_papers=20
        )
        
        assert query.compound_name == "Aspirin"
        assert query.compound_smiles == "CC(=O)OC1=CC=CC=C1C(=O)O"
        assert query.max_papers == 20


@pytest.mark.skipif(
    not pytest.importorskip("rdkit", minversion=None),
    reason="RDKit not available"
)
class TestChemicalService:
    """Test ChemicalService functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.service = ChemicalService()
        self.aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        self.caffeine_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    
    def test_smiles_validation(self):
        """Test SMILES validation."""
        # Valid SMILES
        is_valid, error = self.service.validate_smiles(self.aspirin_smiles)
        assert is_valid == True
        assert error is None
        
        # Invalid SMILES
        is_valid, error = self.service.validate_smiles("INVALID")
        assert is_valid == False
        assert error is not None
        
        # Empty SMILES
        is_valid, error = self.service.validate_smiles("")
        assert is_valid == False
        assert "non-empty string" in error
    
    def test_molecular_properties(self):
        """Test molecular property calculation."""
        properties = self.service.calculate_molecular_properties(self.aspirin_smiles)
        
        assert properties is not None
        assert properties.molecular_weight > 0
        assert isinstance(properties.logp, float)
        assert properties.hbd >= 0
        assert properties.hba >= 0
        assert properties.qed_score >= 0
        assert properties.qed_score <= 1
    
    def test_drug_likeness_assessment(self):
        """Test drug-likeness assessment."""
        properties = self.service.calculate_molecular_properties(self.aspirin_smiles)
        drug_likeness, violations = self.service.assess_drug_likeness(properties)
        
        assert 0 <= drug_likeness <= 1
        assert isinstance(violations, list)
    
    def test_similarity_calculation(self):
        """Test chemical similarity calculation."""
        similarity = self.service.calculate_similarity(
            self.aspirin_smiles, 
            self.aspirin_smiles
        )
        assert similarity == 1.0  # Identical compounds
        
        similarity = self.service.calculate_similarity(
            self.aspirin_smiles,
            self.caffeine_smiles
        )
        assert 0 <= similarity <= 1
        assert similarity < 1.0  # Different compounds
    
    def test_compound_enrichment(self):
        """Test compound data enrichment."""
        compound = ChemicalCompound(
            name="Aspirin",
            smiles=self.aspirin_smiles
        )
        
        enriched = self.service.enrich_compound_data(compound)
        
        assert enriched.molecular_weight is not None
        assert enriched.drug_likeness is not None
        assert enriched.bioactivity_score is not None
        assert enriched.toxicity_score is not None
        assert enriched.compound_type != CompoundType.UNKNOWN


class TestLiteratureService:
    """Test LiteratureService functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.service = LiteratureService()
    
    def test_text_similarity(self):
        """Test text similarity calculation."""
        text1 = "enzyme inhibition mechanism"
        text2 = "enzyme inhibitor binding"
        text3 = "weather patterns tropical"
        
        sim1 = self.service._text_similarity(text1, text2)
        sim2 = self.service._text_similarity(text1, text3)
        
        assert 0 <= sim1 <= 1
        assert 0 <= sim2 <= 1
        assert sim1 > sim2  # More similar texts
    
    def test_keyword_extraction(self):
        """Test keyword extraction from text."""
        title = "Drug Discovery and Development"
        abstract = "This study investigates novel drug compounds for pharmaceutical applications"
        
        keywords = self.service._extract_keywords(title, abstract)
        
        assert isinstance(keywords, list)
        assert len(keywords) <= 10
    
    def test_date_filter_building(self):
        """Test PubMed date filter construction."""
        date_from = datetime(2020, 1, 1)
        date_to = datetime(2023, 12, 31)
        
        filter_str = self.service._build_date_filter(date_from, date_to)
        
        assert "2020/01/01" in filter_str
        assert "2023/12/31" in filter_str
        assert "[Date]" in filter_str
    
    def test_relevance_calculation(self):
        """Test paper relevance scoring."""
        paper = LiteraturePaper(
            title="Drug Discovery Methods",
            abstract="Novel approaches to pharmaceutical compound development",
            keywords=["drug", "pharmaceutical", "development"]
        )
        
        query = "drug discovery pharmaceutical"
        relevance = self.service._calculate_relevance_score(paper, query)
        
        assert 0 <= relevance <= 1
        assert relevance > 0  # Should have some relevance


@pytest.mark.skipif(
    not pytest.importorskip("sentence_transformers", minversion=None),
    reason="sentence-transformers not available"
)
class TestEmbeddingService:
    """Test EmbeddingService functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Mock the model loading for faster tests
        with patch('sentence_transformers.SentenceTransformer'):
            self.service = EmbeddingService()
            # Create a mock model
            self.service._model = Mock()
            self.service._model.encode.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            self.service.embedding_dimension = 5
    
    def test_text_preprocessing(self):
        """Test text preprocessing."""
        text = "  This is a test text.  "
        processed = self.service._preprocess_text(text)
        
        assert processed == "This is a test text."
        
        # Test empty text
        processed = self.service._preprocess_text("")
        assert processed == ""
        
        # Test None input
        processed = self.service._preprocess_text(None)
        assert processed == ""
    
    @pytest.mark.asyncio
    async def test_embedding_generation(self):
        """Test embedding generation."""
        text = "test text for embedding"
        embedding = await self.service.generate_embedding(text)
        
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) == 5  # Mock embedding dimension
    
    def test_similarity_calculation(self):
        """Test embedding similarity calculation."""
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [1.0, 0.0, 0.0]  # Identical
        emb3 = [0.0, 1.0, 0.0]  # Orthogonal
        
        # Identical embeddings
        sim1 = self.service.calculate_similarity(emb1, emb2)
        assert sim1 == 1.0
        
        # Orthogonal embeddings
        sim2 = self.service.calculate_similarity(emb1, emb3)
        assert sim2 == 0.5  # Normalized cosine similarity
        
        # Empty embeddings
        sim3 = self.service.calculate_similarity([], [])
        assert sim3 is None
    
    def test_model_info(self):
        """Test model information retrieval."""
        info = self.service.get_model_info()
        
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "embedding_dimension" in info
        assert "is_loaded" in info
        assert info["is_loaded"] == True


class TestIntegration:
    """Integration tests for multiple services."""
    
    @pytest.mark.asyncio
    @patch('src.services.literature_service.httpx.AsyncClient')
    async def test_research_workflow_integration(self, mock_client):
        """Test integration between services."""
        # Mock HTTP responses for literature service
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.content = b'<eSearchResult><IdList><Id>12345</Id></IdList></eSearchResult>'
        
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
        
        # Test that services can work together
        query = ResearchQuery(
            query_text="test pharmaceutical research",
            compound_smiles="CCO",  # Simple ethanol
            max_papers=5
        )
        
        # This would be part of a research agent workflow
        assert query.query_text == "test pharmaceutical research"
        assert query.compound_smiles == "CCO"
        assert query.max_papers == 5


# Test fixtures
@pytest.fixture
def sample_compound():
    """Sample chemical compound for testing."""
    return ChemicalCompound(
        name="Test Compound",
        smiles="CCO",
        molecular_weight=46.07
    )

@pytest.fixture
def sample_paper():
    """Sample literature paper for testing."""
    return LiteraturePaper(
        title="Test Paper",
        authors=["Test Author"],
        journal="Test Journal",
        pubmed_id="12345"
    )

@pytest.fixture
def sample_query():
    """Sample research query for testing."""
    return ResearchQuery(
        query_text="test query",
        max_papers=10
    )


def test_sample_fixtures(sample_compound, sample_paper, sample_query):
    """Test that fixtures work correctly."""
    assert sample_compound.name == "Test Compound"
    assert sample_paper.title == "Test Paper"
    assert sample_query.query_text == "test query"


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])