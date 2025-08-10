"""
FastAPI application for Pharmaceutical Research Assistant.
Provides REST API endpoints for chemical analysis, literature search, and research workflows.
"""
import logging
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .config import settings
from .models import (
    ChemicalCompound, LiteraturePaper, ResearchQuery, ResearchResult,
    SimilarityResult, APIResponse, HealthCheck, ResearchStatus
)
from .services.chemical_service import ChemicalService
from .services.literature_service import LiteratureService
from .services.embedding_service import EmbeddingService


# Global service instances
chemical_service: Optional[ChemicalService] = None
literature_service: Optional[LiteratureService] = None
embedding_service: Optional[EmbeddingService] = None

# Application startup time
app_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    global chemical_service, literature_service, embedding_service
    
    logging.basicConfig(
        level=getattr(logging, settings.app.log_level),
        format=settings.app.log_format
    )
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Initializing Pharmaceutical Research Assistant services...")
        
        # Initialize services
        chemical_service = ChemicalService()
        literature_service = LiteratureService()
        embedding_service = EmbeddingService()
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise RuntimeError(f"Application startup failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Pharmaceutical Research Assistant")


# Create FastAPI app
app = FastAPI(
    title=settings.app.app_name,
    version=settings.app.version,
    description="AI-powered pharmaceutical research assistant for drug discovery and literature analysis",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.app.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.is_production:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "*.elsevier.com"]
    )


# Dependency injection for services
async def get_chemical_service() -> ChemicalService:
    """Get chemical service instance."""
    if chemical_service is None:
        raise HTTPException(status_code=503, detail="Chemical service not available")
    return chemical_service


async def get_literature_service() -> LiteratureService:
    """Get literature service instance."""
    if literature_service is None:
        raise HTTPException(status_code=503, detail="Literature service not available")
    return literature_service


async def get_embedding_service() -> EmbeddingService:
    """Get embedding service instance."""
    if embedding_service is None:
        raise HTTPException(status_code=503, detail="Embedding service not available")
    return embedding_service


# Utility function for API responses
def create_api_response(
    success: bool,
    message: str,
    data: Any = None,
    errors: List[str] = None,
    execution_time: Optional[float] = None
) -> APIResponse:
    """Create standardized API response."""
    return APIResponse(
        success=success,
        message=message,
        data=data,
        errors=errors or [],
        execution_time_ms=execution_time,
        timestamp=datetime.utcnow()
    )


# Health check endpoint
@app.get("/health", response_model=HealthCheck)
async def health_check():
    """System health check endpoint."""
    try:
        # Check service availability
        chemical_available = chemical_service is not None
        literature_available = literature_service is not None
        embedding_available = embedding_service is not None
        
        # Calculate uptime
        uptime = time.time() - app_start_time
        
        # Overall status
        all_services_available = all([
            chemical_available, 
            literature_available, 
            embedding_available
        ])
        status = "healthy" if all_services_available else "degraded"
        
        return HealthCheck(
            status=status,
            version=settings.app.version,
            database_connected=True,  # SQLite is always available
            redis_connected=False,  # Not implemented in this demo
            embedding_model_loaded=embedding_available,
            uptime_seconds=uptime,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


# Chemical analysis endpoints
@app.post(f"{settings.app.api_prefix}/chemical/analyze", response_model=APIResponse)
async def analyze_chemical_compound(
    compound: ChemicalCompound,
    chemical_svc: ChemicalService = Depends(get_chemical_service)
):
    """
    Analyze a chemical compound and calculate molecular properties.
    """
    start_time = time.time()
    
    try:
        if not compound.smiles:
            raise HTTPException(
                status_code=400, 
                detail="SMILES notation is required for chemical analysis"
            )
        
        # Validate SMILES
        is_valid, error_msg = chemical_svc.validate_smiles(compound.smiles)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid SMILES: {error_msg}")
        
        # Enrich compound with calculated properties
        enriched_compound = chemical_svc.enrich_compound_data(compound)
        
        execution_time = (time.time() - start_time) * 1000
        
        return create_api_response(
            success=True,
            message="Chemical compound analyzed successfully",
            data=enriched_compound.dict(),
            execution_time=execution_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error analyzing compound: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post(f"{settings.app.api_prefix}/chemical/similarity", response_model=APIResponse)
async def find_similar_chemicals(
    query_smiles: str = Query(..., description="Query compound SMILES"),
    candidate_smiles: List[str] = Query(..., description="Candidate compound SMILES list"),
    threshold: float = Query(0.7, ge=0.0, le=1.0, description="Similarity threshold"),
    max_results: int = Query(10, ge=1, le=100, description="Maximum results"),
    chemical_svc: ChemicalService = Depends(get_chemical_service)
):
    """
    Find similar chemical compounds using molecular fingerprint similarity.
    """
    start_time = time.time()
    
    try:
        # Validate query SMILES
        is_valid, error_msg = chemical_svc.validate_smiles(query_smiles)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid query SMILES: {error_msg}")
        
        # Create compound objects for similarity search
        compounds = []
        for i, smiles in enumerate(candidate_smiles):
            compound = ChemicalCompound(
                id=f"compound_{i}",
                name=f"Compound {i+1}",
                smiles=smiles
            )
            compounds.append(compound)
        
        # Find similar compounds
        similar_compounds = chemical_svc.find_similar_compounds(
            query_smiles=query_smiles,
            compound_database=compounds,
            similarity_threshold=threshold,
            max_results=max_results
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        return create_api_response(
            success=True,
            message=f"Found {len(similar_compounds)} similar compounds",
            data=[result.dict() for result in similar_compounds],
            execution_time=execution_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error finding similar chemicals: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Literature search endpoints
@app.get(f"{settings.app.api_prefix}/literature/search", response_model=APIResponse)
async def search_literature(
    query: str = Query(..., description="Search query"),
    max_results: int = Query(20, ge=1, le=100, description="Maximum results"),
    date_from: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    literature_svc: LiteratureService = Depends(get_literature_service)
):
    """
    Search scientific literature using PubMed.
    """
    start_time = time.time()
    
    try:
        # Parse dates
        parsed_date_from = None
        parsed_date_to = None
        
        if date_from:
            parsed_date_from = datetime.strptime(date_from, "%Y-%m-%d")
        if date_to:
            parsed_date_to = datetime.strptime(date_to, "%Y-%m-%d")
        
        # Search literature
        papers = await literature_svc.search_pubmed(
            query=query,
            max_results=max_results,
            date_from=parsed_date_from,
            date_to=parsed_date_to
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        return create_api_response(
            success=True,
            message=f"Found {len(papers)} research papers",
            data=[paper.dict() for paper in papers],
            execution_time=execution_time
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        logging.error(f"Error searching literature: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get(f"{settings.app.api_prefix}/literature/trending", response_model=APIResponse)
async def get_trending_topics(
    days: int = Query(30, ge=1, le=365, description="Days to look back"),
    literature_svc: LiteratureService = Depends(get_literature_service)
):
    """
    Get trending topics in pharmaceutical research.
    """
    start_time = time.time()
    
    try:
        trending_topics = await literature_svc.get_trending_topics(days=days)
        
        execution_time = (time.time() - start_time) * 1000
        
        return create_api_response(
            success=True,
            message=f"Found {len(trending_topics)} trending topics",
            data={"topics": trending_topics, "period_days": days},
            execution_time=execution_time
        )
        
    except Exception as e:
        logging.error(f"Error getting trending topics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Semantic search endpoints
@app.post(f"{settings.app.api_prefix}/semantic/similarity", response_model=APIResponse)
async def semantic_similarity_search(
    query_text: str = Query(..., description="Query text"),
    candidate_texts: List[str] = Query(..., description="Candidate texts"),
    threshold: float = Query(0.6, ge=0.0, le=1.0, description="Similarity threshold"),
    max_results: int = Query(10, ge=1, le=100, description="Maximum results"),
    embedding_svc: EmbeddingService = Depends(get_embedding_service)
):
    """
    Find semantically similar texts using embeddings.
    """
    start_time = time.time()
    
    try:
        similar_texts = await embedding_svc.find_similar_texts(
            query_text=query_text,
            candidate_texts=candidate_texts,
            similarity_threshold=threshold,
            max_results=max_results
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        return create_api_response(
            success=True,
            message=f"Found {len(similar_texts)} similar texts",
            data=[result.dict() for result in similar_texts],
            execution_time=execution_time
        )
        
    except Exception as e:
        logging.error(f"Error in semantic similarity search: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Comprehensive research endpoint
@app.post(f"{settings.app.api_prefix}/research/comprehensive", response_model=APIResponse)
async def comprehensive_research(
    research_query: ResearchQuery,
    background_tasks: BackgroundTasks,
    chemical_svc: ChemicalService = Depends(get_chemical_service),
    literature_svc: LiteratureService = Depends(get_literature_service),
    embedding_svc: EmbeddingService = Depends(get_embedding_service)
):
    """
    Perform comprehensive pharmaceutical research combining multiple services.
    """
    start_time = time.time()
    
    try:
        # Initialize research result
        result = ResearchResult(
            query_id=f"research_{int(time.time())}",
            status=ResearchStatus.IN_PROGRESS
        )
        
        # Literature search
        if research_query.include_literature_mining:
            papers = await literature_svc.search_pubmed(
                query=research_query.query_text,
                max_results=research_query.max_papers,
                date_from=research_query.date_from,
                date_to=research_query.date_to
            )
            result.literature_papers = papers
        
        # Chemical similarity search
        if research_query.include_chemical_similarity and research_query.compound_smiles:
            # For demo, we'll use a small database of known compounds
            sample_compounds = [
                ChemicalCompound(
                    id="aspirin",
                    name="Aspirin",
                    smiles="CC(=O)OC1=CC=CC=C1C(=O)O"
                ),
                ChemicalCompound(
                    id="caffeine",
                    name="Caffeine",
                    smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
                ),
                ChemicalCompound(
                    id="ibuprofen",
                    name="Ibuprofen",
                    smiles="CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
                )
            ]
            
            similar_compounds = chemical_svc.find_similar_compounds(
                query_smiles=research_query.compound_smiles,
                compound_database=sample_compounds,
                similarity_threshold=0.3,
                max_results=10
            )
            result.similar_compounds = similar_compounds
        
        # Generate key insights
        insights = []
        if result.literature_papers:
            insights.append(f"Found {len(result.literature_papers)} relevant research papers")
            
            # Extract common keywords
            all_keywords = []
            for paper in result.literature_papers:
                all_keywords.extend(paper.keywords)
            
            if all_keywords:
                keyword_counts = {}
                for keyword in all_keywords:
                    keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
                
                top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                insights.append(f"Key research areas: {', '.join([kw[0] for kw in top_keywords])}")
        
        if result.similar_compounds:
            insights.append(f"Identified {len(result.similar_compounds)} structurally similar compounds")
        
        result.key_insights = insights
        
        # Calculate confidence score
        confidence = 0.7  # Base confidence
        if result.literature_papers:
            confidence += 0.2
        if result.similar_compounds:
            confidence += 0.1
        result.confidence_score = min(1.0, confidence)
        
        # Mark as completed
        result.status = ResearchStatus.COMPLETED
        result.completed_at = datetime.utcnow()
        
        execution_time = (time.time() - start_time) * 1000
        result.processing_time_seconds = execution_time / 1000
        
        return create_api_response(
            success=True,
            message="Comprehensive research completed successfully",
            data=result.dict(),
            execution_time=execution_time
        )
        
    except Exception as e:
        logging.error(f"Error in comprehensive research: {e}")
        
        # Mark as failed
        result.status = ResearchStatus.FAILED
        result.errors = [str(e)]
        
        raise HTTPException(status_code=500, detail="Research processing failed")


# Service info endpoints
@app.get(f"{settings.app.api_prefix}/info/embedding-model", response_model=APIResponse)
async def get_embedding_model_info(
    embedding_svc: EmbeddingService = Depends(get_embedding_service)
):
    """Get information about the loaded embedding model."""
    try:
        model_info = embedding_svc.get_model_info()
        
        return create_api_response(
            success=True,
            message="Embedding model information retrieved",
            data=model_info
        )
        
    except Exception as e:
        logging.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Custom exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logging.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=create_api_response(
            success=False,
            message="Internal server error",
            errors=[str(exc)]
        ).dict()
    )


# Development server runner
if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.app.debug,
        log_level=settings.app.log_level.lower()
    )