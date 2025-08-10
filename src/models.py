"""
Pydantic models for pharmaceutical research assistant API.
Defines data structures for chemical compounds, literature, and research workflows.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator


class ResearchStatus(str, Enum):
    """Research workflow status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class CompoundType(str, Enum):
    """Chemical compound type classification."""
    SMALL_MOLECULE = "small_molecule"
    PROTEIN = "protein"
    PEPTIDE = "peptide"
    NUCLEIC_ACID = "nucleic_acid"
    UNKNOWN = "unknown"


class ChemicalCompound(BaseModel):
    """Chemical compound data model."""
    
    id: Optional[str] = Field(None, description="Unique compound identifier")
    name: str = Field(..., description="Compound name")
    smiles: Optional[str] = Field(None, description="SMILES notation")
    inchi: Optional[str] = Field(None, description="InChI notation")
    molecular_formula: Optional[str] = Field(None, description="Molecular formula")
    molecular_weight: Optional[float] = Field(None, description="Molecular weight in g/mol")
    compound_type: CompoundType = Field(
        default=CompoundType.UNKNOWN, 
        description="Type of chemical compound"
    )
    
    # Pharmacological properties
    bioactivity_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Predicted bioactivity score (0-1)"
    )
    drug_likeness: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Drug-likeness score (0-1)"
    )
    toxicity_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Predicted toxicity score (0-1)"
    )
    
    # Metadata
    source: Optional[str] = Field(None, description="Data source")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('smiles')
    def validate_smiles(cls, v):
        """Validate SMILES notation format."""
        if v and len(v.strip()) == 0:
            return None
        return v


class LiteraturePaper(BaseModel):
    """Scientific literature paper model."""
    
    pubmed_id: Optional[str] = Field(None, description="PubMed ID")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    title: str = Field(..., description="Paper title")
    abstract: Optional[str] = Field(None, description="Paper abstract")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    journal: Optional[str] = Field(None, description="Journal name")
    publication_date: Optional[datetime] = Field(None, description="Publication date")
    
    # Content analysis
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    chemical_entities: List[str] = Field(
        default_factory=list, 
        description="Mentioned chemical entities"
    )
    relevance_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Relevance score for query"
    )
    
    # Metadata
    source: str = Field(default="pubmed", description="Literature source")
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)


class SimilarityResult(BaseModel):
    """Similarity search result model."""
    
    target_id: str = Field(..., description="Target compound/document ID")
    similarity_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Similarity score (0-1)"
    )
    similarity_type: str = Field(..., description="Type of similarity calculation")
    target_data: Optional[Dict[str, Any]] = Field(
        None, 
        description="Additional target data"
    )


class ResearchQuery(BaseModel):
    """Research query input model."""
    
    query_text: str = Field(..., description="Natural language research query")
    compound_smiles: Optional[str] = Field(None, description="SMILES for compound focus")
    compound_name: Optional[str] = Field(None, description="Compound name for focus")
    
    # Search parameters
    max_papers: int = Field(
        default=20, 
        ge=1, 
        le=100, 
        description="Maximum number of papers to retrieve"
    )
    include_chemical_similarity: bool = Field(
        default=True, 
        description="Include chemical similarity search"
    )
    include_literature_mining: bool = Field(
        default=True, 
        description="Include literature mining"
    )
    
    # Filters
    date_from: Optional[datetime] = Field(None, description="Search from date")
    date_to: Optional[datetime] = Field(None, description="Search to date")
    journals: Optional[List[str]] = Field(None, description="Specific journals to search")


class ResearchResult(BaseModel):
    """Research workflow result model."""
    
    query_id: str = Field(..., description="Unique query identifier")
    status: ResearchStatus = Field(..., description="Research status")
    
    # Results
    chemical_compounds: List[ChemicalCompound] = Field(
        default_factory=list, 
        description="Found chemical compounds"
    )
    literature_papers: List[LiteraturePaper] = Field(
        default_factory=list, 
        description="Retrieved literature papers"
    )
    similar_compounds: List[SimilarityResult] = Field(
        default_factory=list, 
        description="Similar compounds found"
    )
    
    # Analysis
    key_insights: List[str] = Field(
        default_factory=list, 
        description="Key research insights"
    )
    compound_interactions: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Predicted compound interactions"
    )
    research_gaps: List[str] = Field(
        default_factory=list, 
        description="Identified research gaps"
    )
    
    # Metadata
    processing_time_seconds: Optional[float] = Field(
        None, 
        description="Total processing time"
    )
    confidence_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Overall confidence in results"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    
    # Error handling
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")


class EmbeddingVector(BaseModel):
    """Vector embedding model."""
    
    id: str = Field(..., description="Unique embedding identifier")
    text: str = Field(..., description="Original text")
    vector: List[float] = Field(..., description="Embedding vector")
    model_name: str = Field(..., description="Embedding model used")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class APIResponse(BaseModel):
    """Standard API response wrapper."""
    
    success: bool = Field(..., description="Request success status")
    message: str = Field(..., description="Response message")
    data: Optional[Union[Dict[str, Any], List[Any]]] = Field(
        None, 
        description="Response data"
    )
    errors: List[str] = Field(default_factory=list, description="Error messages")
    execution_time_ms: Optional[float] = Field(
        None, 
        description="Execution time in milliseconds"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthCheck(BaseModel):
    """System health check model."""
    
    status: str = Field(..., description="System status")
    version: str = Field(..., description="Application version")
    database_connected: bool = Field(..., description="Database connection status")
    redis_connected: bool = Field(..., description="Redis connection status")
    embedding_model_loaded: bool = Field(..., description="Embedding model status")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)