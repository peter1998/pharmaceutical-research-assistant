"""
API integration tests for pharmaceutical research assistant.
Tests HTTP endpoints and service integration.
"""
import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "version" in data
    assert "uptime_seconds" in data


def test_chemical_analyze_endpoint(client):
    """Test chemical analysis endpoint."""
    compound_data = {
        "name": "Aspirin",
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"
    }
    
    response = client.post("/api/v1/chemical/analyze", json=compound_data)
    assert response.status_code == 200
    
    result = response.json()
    assert result["success"] == True
    assert "data" in result
    assert result["data"]["name"] == "Aspirin"


def test_similarity_search_endpoint(client):
    """Test chemical similarity endpoint."""
    params = {
        "query_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "candidate_smiles": ["CCO", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"],
        "threshold": 0.3
    }
    
    response = client.post("/api/v1/chemical/similarity", params=params)
    assert response.status_code == 200
    
    result = response.json()
    assert result["success"] == True
    assert "data" in result


@pytest.mark.asyncio
async def test_literature_search_endpoint():
    """Test literature search endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/literature/search",
            params={"query": "aspirin", "max_results": 5}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] == True


def test_semantic_similarity_endpoint(client):
    """Test semantic similarity search."""
    data = {
        "query_text": "drug discovery",
        "candidate_texts": ["pharmaceutical research", "weather forecast"],
        "threshold": 0.5
    }
    
    response = client.post("/api/v1/semantic/similarity", json=data)
    assert response.status_code == 200
    
    result = response.json()
    assert result["success"] == True


def test_comprehensive_research_endpoint(client):
    """Test comprehensive research workflow."""
    research_data = {
        "query_text": "anti-inflammatory drugs",
        "compound_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "compound_name": "Aspirin",
        "max_papers": 5,
        "include_chemical_similarity": True,
        "include_literature_mining": True
    }
    
    response = client.post("/api/v1/research/comprehensive", json=research_data)
    assert response.status_code == 200
    
    result = response.json()
    assert result["success"] == True
    assert "data" in result


def test_embedding_model_info_endpoint(client):
    """Test embedding model info endpoint."""
    response = client.get("/api/v1/info/embedding-model")
    assert response.status_code == 200
    
    result = response.json()
    assert result["success"] == True
    assert "model_name" in result["data"]


def test_invalid_smiles_handling(client):
    """Test error handling for invalid SMILES."""
    compound_data = {
        "name": "Invalid",
        "smiles": "INVALID_SMILES"
    }
    
    response = client.post("/api/v1/chemical/analyze", json=compound_data)
    assert response.status_code == 400
    
    assert "Invalid SMILES" in response.json()["detail"]


def test_missing_required_fields(client):
    """Test validation for missing required fields."""
    # Missing name field
    response = client.post("/api/v1/chemical/analyze", json={"smiles": "CCO"})
    assert response.status_code == 422  # Validation error


def test_api_response_format(client):
    """Test that all API responses follow standard format."""
    response = client.get("/health")
    assert response.status_code == 200
    
    # Health check has different format, test regular API endpoint
    compound_data = {
        "name": "Test",
        "smiles": "CCO"
    }
    
    response = client.post("/api/v1/chemical/analyze", json=compound_data)
    assert response.status_code == 200
    
    result = response.json()
    
    # Check standard API response format
    assert "success" in result
    assert "message" in result
    assert "data" in result
    assert "timestamp" in result
    assert "execution_time_ms" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])