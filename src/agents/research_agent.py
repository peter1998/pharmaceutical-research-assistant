"""
Pharmaceutical Research Agent - orchestrates multi-modal research workflows.
Combines chemical analysis, literature mining, and semantic search for comprehensive drug discovery.
"""
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

from ..models import (
    ResearchQuery, ResearchResult, ResearchStatus, ChemicalCompound,
    LiteraturePaper, SimilarityResult
)
from ..services.chemical_service import ChemicalService
from ..services.literature_service import LiteratureService
from ..services.embedding_service import EmbeddingService


@dataclass
class ResearchContext:
    """Context information for research workflow."""
    query: ResearchQuery
    start_time: datetime = field(default_factory=datetime.utcnow)
    chemicals_found: List[ChemicalCompound] = field(default_factory=list)
    papers_found: List[LiteraturePaper] = field(default_factory=list)
    similar_compounds: List[SimilarityResult] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class PharmaceuticalResearchAgent:
    """
    Main research agent that orchestrates pharmaceutical research workflows.
    """
    
    def __init__(
        self,
        chemical_service: ChemicalService,
        literature_service: LiteratureService,
        embedding_service: EmbeddingService
    ):
        self.logger = logging.getLogger(__name__)
        self.chemical_service = chemical_service
        self.literature_service = literature_service
        self.embedding_service = embedding_service
        
        # Research workflow configuration
        self.max_concurrent_searches = 3
        self.similarity_threshold = 0.7
        self.confidence_threshold = 0.6
        
    async def execute_research(self, query: ResearchQuery) -> ResearchResult:
        """
        Execute comprehensive pharmaceutical research workflow.
        
        Args:
            query: Research query with parameters
            
        Returns:
            Complete research results
        """
        # Initialize research context
        context = ResearchContext(query=query)
        
        # Create result object
        result = ResearchResult(
            query_id=f"research_{int(datetime.utcnow().timestamp())}",
            status=ResearchStatus.IN_PROGRESS
        )
        
        try:
            self.logger.info(f"Starting research for query: {query.query_text}")
            
            # Phase 1: Literature Discovery
            if query.include_literature_mining:
                await self._literature_discovery_phase(context)
                result.literature_papers = context.papers_found
            
            # Phase 2: Chemical Analysis
            if query.include_chemical_similarity:
                await self._chemical_analysis_phase(context)
                result.chemical_compounds = context.chemicals_found
                result.similar_compounds = context.similar_compounds
            
            # Phase 3: Cross-Modal Analysis
            await self._cross_modal_analysis_phase(context)
            
            # Phase 4: Insight Generation
            await self._insight_generation_phase(context)
            result.key_insights = context.insights
            
            # Phase 5: Research Gap Analysis
            research_gaps = await self._identify_research_gaps(context)
            result.research_gaps = research_gaps
            
            # Calculate final metrics
            processing_time = (datetime.utcnow() - context.start_time).total_seconds()
            result.processing_time_seconds = processing_time
            result.confidence_score = self._calculate_confidence_score(context)
            
            # Set completion status
            result.status = ResearchStatus.COMPLETED
            result.completed_at = datetime.utcnow()
            result.warnings = context.warnings
            
            self.logger.info(
                f"Research completed in {processing_time:.2f}s. "
                f"Found {len(result.literature_papers)} papers, "
                f"{len(result.chemical_compounds)} compounds"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Research execution failed: {e}")
            
            result.status = ResearchStatus.FAILED
            result.errors = context.errors + [str(e)]
            result.completed_at = datetime.utcnow()
            
            return result
    
    async def _literature_discovery_phase(self, context: ResearchContext):
        """
        Phase 1: Discover relevant literature using multiple search strategies.
        """
        self.logger.info("Phase 1: Literature Discovery")
        
        try:
            # Primary literature search
            papers = await self.literature_service.search_pubmed(
                query=context.query.query_text,
                max_results=context.query.max_papers,
                date_from=context.query.date_from,
                date_to=context.query.date_to
            )
            
            if papers:
                context.papers_found.extend(papers)
                self.logger.info(f"Found {len(papers)} papers from primary search")
            else:
                context.warnings.append("No papers found in primary literature search")
            
            # If we have a specific compound, search for it
            if context.query.compound_name:
                compound_papers = await self.literature_service.search_pubmed(
                    query=f"{context.query.compound_name} drug",
                    max_results=10,
                    date_from=context.query.date_from,
                    date_to=context.query.date_to
                )
                
                # Merge results, avoiding duplicates
                existing_pmids = {p.pubmed_id for p in context.papers_found if p.pubmed_id}
                new_papers = [p for p in compound_papers if p.pubmed_id not in existing_pmids]
                context.papers_found.extend(new_papers)
                
                if new_papers:
                    self.logger.info(f"Found {len(new_papers)} additional papers for compound")
            
        except Exception as e:
            error_msg = f"Literature discovery failed: {e}"
            self.logger.error(error_msg)
            context.errors.append(error_msg)
    
    async def _chemical_analysis_phase(self, context: ResearchContext):
        """
        Phase 2: Analyze chemical compounds and find similarities.
        """
        self.logger.info("Phase 2: Chemical Analysis")
        
        try:
            # If SMILES is provided, analyze the query compound
            if context.query.compound_smiles:
                await self._analyze_query_compound(context)
            
            # Extract compounds from literature
            if context.papers_found:
                await self._extract_compounds_from_literature(context)
            
            # Find similar compounds if we have a reference
            if context.query.compound_smiles and context.chemicals_found:
                await self._find_similar_compounds(context)
            
        except Exception as e:
            error_msg = f"Chemical analysis failed: {e}"
            self.logger.error(error_msg)
            context.errors.append(error_msg)
    
    async def _analyze_query_compound(self, context: ResearchContext):
        """Analyze the query compound's properties."""
        try:
            # Create compound object
            query_compound = ChemicalCompound(
                id="query_compound",
                name=context.query.compound_name or "Query Compound",
                smiles=context.query.compound_smiles
            )
            
            # Validate SMILES
            is_valid, error_msg = self.chemical_service.validate_smiles(query_compound.smiles)
            if not is_valid:
                context.errors.append(f"Invalid query SMILES: {error_msg}")
                return
            
            # Enrich with calculated properties
            enriched_compound = self.chemical_service.enrich_compound_data(query_compound)
            context.chemicals_found.append(enriched_compound)
            
            # Generate insights about the compound
            if enriched_compound.drug_likeness:
                if enriched_compound.drug_likeness > 0.7:
                    context.insights.append("Query compound shows high drug-likeness potential")
                elif enriched_compound.drug_likeness < 0.3:
                    context.insights.append("Query compound may have drug-likeness issues")
            
            self.logger.info(f"Analyzed query compound: {enriched_compound.name}")
            
        except Exception as e:
            self.logger.error(f"Query compound analysis failed: {e}")
            context.warnings.append("Could not analyze query compound properties")
    
    async def _extract_compounds_from_literature(self, context: ResearchContext):
        """Extract chemical compounds mentioned in literature."""
        try:
            compound_mentions = set()
            
            # Simple extraction of compound names from papers
            for paper in context.papers_found:
                # Extract from chemical entities found in papers
                compound_mentions.update(paper.chemical_entities)
                
                # Simple pattern matching for compound names in titles/abstracts
                text = paper.title.lower()
                if paper.abstract:
                    text += " " + paper.abstract.lower()
                
                # Look for compound-related keywords
                compound_keywords = [
                    "aspirin", "ibuprofen", "acetaminophen", "caffeine",
                    "morphine", "codeine", "penicillin", "insulin"
                ]
                
                for keyword in compound_keywords:
                    if keyword in text:
                        compound_mentions.add(keyword.title())
            
            # Create compound objects for mentions
            for mention in list(compound_mentions)[:5]:  # Limit to 5 for demo
                compound = ChemicalCompound(
                    id=f"literature_{mention.lower()}",
                    name=mention,
                    source="literature_extraction"
                )
                context.chemicals_found.append(compound)
            
            if compound_mentions:
                context.insights.append(f"Extracted {len(compound_mentions)} chemical compounds from literature")
            
        except Exception as e:
            self.logger.error(f"Compound extraction failed: {e}")
            context.warnings.append("Could not extract compounds from literature")
    
    async def _find_similar_compounds(self, context: ResearchContext):
        """Find compounds similar to the query compound."""
        try:
            if not context.query.compound_smiles:
                return
            
            # Create a database of known pharmaceutical compounds for similarity search
            known_compounds = [
                ChemicalCompound(
                    id="aspirin",
                    name="Aspirin",
                    smiles="CC(=O)OC1=CC=CC=C1C(=O)O"
                ),
                ChemicalCompound(
                    id="ibuprofen",
                    name="Ibuprofen",
                    smiles="CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
                ),
                ChemicalCompound(
                    id="acetaminophen",
                    name="Acetaminophen",
                    smiles="CC(=O)NC1=CC=C(C=C1)O"
                ),
                ChemicalCompound(
                    id="caffeine",
                    name="Caffeine",
                    smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
                ),
                ChemicalCompound(
                    id="morphine",
                    name="Morphine",
                    smiles="CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"
                )
            ]
            
            # Find similar compounds
            similar_compounds = self.chemical_service.find_similar_compounds(
                query_smiles=context.query.compound_smiles,
                compound_database=known_compounds,
                similarity_threshold=0.3,  # Lower threshold for demo
                max_results=5
            )
            
            context.similar_compounds.extend(similar_compounds)
            
            if similar_compounds:
                top_similarity = max(r.similarity_score for r in similar_compounds)
                context.insights.append(
                    f"Found {len(similar_compounds)} similar compounds "
                    f"(top similarity: {top_similarity:.2f})"
                )
            
        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            context.warnings.append("Could not perform chemical similarity search")
    
    async def _cross_modal_analysis_phase(self, context: ResearchContext):
        """
        Phase 3: Cross-modal analysis combining literature and chemical data.
        """
        self.logger.info("Phase 3: Cross-Modal Analysis")
        
        try:
            # Semantic similarity between query and papers
            if context.papers_found:
                await self._semantic_literature_analysis(context)
            
            # Compound-literature relevance mapping
            if context.chemicals_found and context.papers_found:
                await self._map_compounds_to_literature(context)
            
        except Exception as e:
            error_msg = f"Cross-modal analysis failed: {e}"
            self.logger.error(error_msg)
            context.errors.append(error_msg)
    
    async def _semantic_literature_analysis(self, context: ResearchContext):
        """Perform semantic analysis of literature papers."""
        try:
            # Find papers most semantically similar to the query
            similar_papers = await self.embedding_service.find_similar_papers(
                query_text=context.query.query_text,
                papers=context.papers_found,
                similarity_threshold=0.5,
                max_results=5
            )
            
            if similar_papers:
                avg_similarity = sum(r.similarity_score for r in similar_papers) / len(similar_papers)
                context.insights.append(
                    f"Average semantic similarity to query: {avg_similarity:.2f}"
                )
                
                # Identify the most relevant paper
                top_paper = similar_papers[0]
                context.insights.append(
                    f"Most relevant paper: '{top_paper.target_data.get('title', 'Unknown')}' "
                    f"(similarity: {top_paper.similarity_score:.2f})"
                )
            
        except Exception as e:
            self.logger.error(f"Semantic analysis failed: {e}")
            context.warnings.append("Could not perform semantic literature analysis")
    
    async def _map_compounds_to_literature(self, context: ResearchContext):
        """Map chemical compounds to relevant literature."""
        try:
            # For each compound, find papers that mention it
            compound_paper_mapping = {}
            
            for compound in context.chemicals_found:
                relevant_papers = []
                
                for paper in context.papers_found:
                    # Check if compound name appears in paper
                    title_text = paper.title.lower()
                    abstract_text = paper.abstract.lower() if paper.abstract else ""
                    full_text = title_text + " " + abstract_text
                    
                    if compound.name.lower() in full_text:
                        relevant_papers.append(paper)
                
                if relevant_papers:
                    compound_paper_mapping[compound.name] = len(relevant_papers)
            
            if compound_paper_mapping:
                most_studied = max(compound_paper_mapping.items(), key=lambda x: x[1])
                context.insights.append(
                    f"Most studied compound: {most_studied[0]} "
                    f"({most_studied[1]} papers)"
                )
            
        except Exception as e:
            self.logger.error(f"Compound-literature mapping failed: {e}")
            context.warnings.append("Could not map compounds to literature")
    
    async def _insight_generation_phase(self, context: ResearchContext):
        """
        Phase 4: Generate research insights and recommendations.
        """
        self.logger.info("Phase 4: Insight Generation")
        
        try:
            # Research volume insights
            if context.papers_found:
                recent_papers = [
                    p for p in context.papers_found 
                    if p.publication_date and 
                    (datetime.utcnow() - p.publication_date).days <= 365
                ]
                
                if recent_papers:
                    context.insights.append(
                        f"{len(recent_papers)} papers published in the last year "
                        f"({len(recent_papers)/len(context.papers_found)*100:.1f}% of total)"
                    )
            
            # Drug-likeness insights
            drug_like_compounds = [
                c for c in context.chemicals_found 
                if c.drug_likeness and c.drug_likeness > 0.7
            ]
            
            if drug_like_compounds:
                context.insights.append(
                    f"{len(drug_like_compounds)} compounds show high drug-likeness potential"
                )
            
            # Safety insights
            low_toxicity_compounds = [
                c for c in context.chemicals_found 
                if c.toxicity_score and c.toxicity_score < 0.3
            ]
            
            if low_toxicity_compounds:
                context.insights.append(
                    f"{len(low_toxicity_compounds)} compounds show low predicted toxicity"
                )
            
            # Research trends
            if len(context.papers_found) > 5:
                # Simple trend analysis based on publication years
                pub_years = [
                    p.publication_date.year for p in context.papers_found 
                    if p.publication_date
                ]
                
                if pub_years:
                    year_counts = {}
                    for year in pub_years:
                        year_counts[year] = year_counts.get(year, 0) + 1
                    
                    recent_years = [y for y in pub_years if y >= 2020]
                    if len(recent_years) > len(pub_years) * 0.6:
                        context.insights.append("Research interest appears to be increasing recently")
            
        except Exception as e:
            self.logger.error(f"Insight generation failed: {e}")
            context.warnings.append("Could not generate comprehensive insights")
    
    async def _identify_research_gaps(self, context: ResearchContext) -> List[str]:
        """Identify potential research gaps and opportunities."""
        gaps = []
        
        try:
            # Check for lack of recent research
            if context.papers_found:
                recent_papers = [
                    p for p in context.papers_found 
                    if p.publication_date and 
                    (datetime.utcnow() - p.publication_date).days <= 730  # 2 years
                ]
                
                if len(recent_papers) < len(context.papers_found) * 0.3:
                    gaps.append("Limited recent research activity in this area")
            
            # Check for lack of clinical trial data
            clinical_papers = [
                p for p in context.papers_found 
                if any(keyword in p.title.lower() for keyword in ['clinical trial', 'phase', 'efficacy'])
            ]
            
            if len(clinical_papers) < len(context.papers_found) * 0.2:
                gaps.append("Limited clinical trial data available")
            
            # Check for lack of safety data
            safety_papers = [
                p for p in context.papers_found 
                if any(keyword in p.title.lower() for keyword in ['safety', 'toxicity', 'adverse'])
            ]
            
            if len(safety_papers) < len(context.papers_found) * 0.1:
                gaps.append("Limited safety and toxicity studies")
            
            # Check for compound diversity
            if len(context.chemicals_found) < 3:
                gaps.append("Limited chemical compound diversity in current research")
            
        except Exception as e:
            self.logger.error(f"Research gap analysis failed: {e}")
            gaps.append("Could not perform comprehensive gap analysis")
        
        return gaps
    
    def _calculate_confidence_score(self, context: ResearchContext) -> float:
        """Calculate overall confidence score for the research results."""
        try:
            score = 0.5  # Base confidence
            
            # Literature coverage
            if len(context.papers_found) >= 10:
                score += 0.2
            elif len(context.papers_found) >= 5:
                score += 0.1
            
            # Chemical analysis quality
            if context.chemicals_found:
                analyzed_compounds = [
                    c for c in context.chemicals_found 
                    if c.drug_likeness is not None
                ]
                if analyzed_compounds:
                    score += 0.15
            
            # Similarity search success
            if context.similar_compounds:
                high_sim = [s for s in context.similar_compounds if s.similarity_score > 0.7]
                if high_sim:
                    score += 0.1
            
            # Insight generation
            if len(context.insights) >= 3:
                score += 0.1
            
            # Error penalty
            if context.errors:
                score -= 0.1 * len(context.errors)
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5  # Default confidence if calculation fails