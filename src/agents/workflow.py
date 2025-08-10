"""
Workflow orchestration system for pharmaceutical research.
Coordinates multi-step research processes with state management and error recovery.
"""
import logging
import asyncio
import uuid
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

from ..models import (
    ResearchQuery, ResearchResult, ResearchStatus, ChemicalCompound,
    LiteraturePaper, SimilarityResult
)
from ..services.chemical_service import ChemicalService
from ..services.literature_service import LiteratureService
from ..services.embedding_service import EmbeddingService


class WorkflowState(str, Enum):
    """Workflow execution states."""
    PENDING = "pending"
    INITIALIZING = "initializing"
    CHEMICAL_ANALYSIS = "chemical_analysis"
    LITERATURE_SEARCH = "literature_search"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    CROSS_MODAL_ANALYSIS = "cross_modal_analysis"
    INSIGHT_GENERATION = "insight_generation"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowTransition(str, Enum):
    """Valid workflow state transitions."""
    START = "start"
    PROCEED = "proceed"
    RETRY = "retry"
    SKIP = "skip"
    FAIL = "fail"
    CANCEL = "cancel"
    COMPLETE = "complete"


@dataclass
class WorkflowStep:
    """Individual workflow step definition."""
    name: str
    state: WorkflowState
    handler: Callable
    timeout_seconds: int = 300
    retry_count: int = 3
    retry_delay_seconds: int = 5
    required: bool = True
    dependencies: List[WorkflowState] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.dependencies:
            self.dependencies = []


@dataclass
class WorkflowContext:
    """Workflow execution context and state."""
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: Optional[ResearchQuery] = None
    current_state: WorkflowState = WorkflowState.PENDING
    previous_state: Optional[WorkflowState] = None
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    
    # Research data
    chemical_compounds: List[ChemicalCompound] = field(default_factory=list)
    literature_papers: List[LiteraturePaper] = field(default_factory=list)
    similar_compounds: List[SimilarityResult] = field(default_factory=list)
    semantic_results: List[SimilarityResult] = field(default_factory=list)
    
    # Step execution tracking
    step_results: Dict[str, Any] = field(default_factory=dict)
    step_errors: Dict[str, List[str]] = field(default_factory=dict)
    step_timings: Dict[str, float] = field(default_factory=dict)
    
    # Workflow metadata
    insights: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    confidence_score: float = 0.5
    
    @property
    def duration(self) -> Optional[float]:
        """Total workflow duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def is_complete(self) -> bool:
        """Check if workflow is in a terminal state."""
        return self.current_state in [
            WorkflowState.COMPLETED,
            WorkflowState.FAILED,
            WorkflowState.CANCELLED
        ]


class WorkflowEngine:
    """
    Workflow execution engine for pharmaceutical research workflows.
    Provides state management, error recovery, and step coordination.
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
        
        # Active workflow tracking
        self.active_workflows: Dict[str, WorkflowContext] = {}
        
        # Workflow step definitions
        self.workflow_steps = self._define_workflow_steps()
        
        # State transition rules
        self.state_transitions = self._define_state_transitions()
    
    def _define_workflow_steps(self) -> Dict[WorkflowState, WorkflowStep]:
        """Define all workflow steps and their configurations."""
        return {
            WorkflowState.INITIALIZING: WorkflowStep(
                name="Initialize Workflow",
                state=WorkflowState.INITIALIZING,
                handler=self._initialize_workflow,
                timeout_seconds=30,
                retry_count=1
            ),
            
            WorkflowState.CHEMICAL_ANALYSIS: WorkflowStep(
                name="Chemical Analysis",
                state=WorkflowState.CHEMICAL_ANALYSIS,
                handler=self._execute_chemical_analysis,
                timeout_seconds=120,
                retry_count=3,
                dependencies=[WorkflowState.INITIALIZING]
            ),
            
            WorkflowState.LITERATURE_SEARCH: WorkflowStep(
                name="Literature Search",
                state=WorkflowState.LITERATURE_SEARCH,
                handler=self._execute_literature_search,
                timeout_seconds=300,
                retry_count=2,
                dependencies=[WorkflowState.INITIALIZING]
            ),
            
            WorkflowState.SEMANTIC_ANALYSIS: WorkflowStep(
                name="Semantic Analysis",
                state=WorkflowState.SEMANTIC_ANALYSIS,
                handler=self._execute_semantic_analysis,
                timeout_seconds=180,
                retry_count=2,
                dependencies=[WorkflowState.LITERATURE_SEARCH]
            ),
            
            WorkflowState.CROSS_MODAL_ANALYSIS: WorkflowStep(
                name="Cross-Modal Analysis",
                state=WorkflowState.CROSS_MODAL_ANALYSIS,
                handler=self._execute_cross_modal_analysis,
                timeout_seconds=120,
                retry_count=2,
                dependencies=[WorkflowState.CHEMICAL_ANALYSIS, WorkflowState.LITERATURE_SEARCH]
            ),
            
            WorkflowState.INSIGHT_GENERATION: WorkflowStep(
                name="Insight Generation",
                state=WorkflowState.INSIGHT_GENERATION,
                handler=self._generate_insights,
                timeout_seconds=60,
                retry_count=1,
                dependencies=[WorkflowState.CROSS_MODAL_ANALYSIS]
            ),
            
            WorkflowState.FINALIZING: WorkflowStep(
                name="Finalize Results",
                state=WorkflowState.FINALIZING,
                handler=self._finalize_workflow,
                timeout_seconds=30,
                retry_count=1,
                dependencies=[WorkflowState.INSIGHT_GENERATION]
            )
        }
    
    def _define_state_transitions(self) -> Dict[WorkflowState, List[WorkflowState]]:
        """Define valid state transitions."""
        return {
            WorkflowState.PENDING: [WorkflowState.INITIALIZING, WorkflowState.CANCELLED],
            WorkflowState.INITIALIZING: [
                WorkflowState.CHEMICAL_ANALYSIS,
                WorkflowState.LITERATURE_SEARCH,
                WorkflowState.FAILED
            ],
            WorkflowState.CHEMICAL_ANALYSIS: [
                WorkflowState.LITERATURE_SEARCH,
                WorkflowState.CROSS_MODAL_ANALYSIS,
                WorkflowState.FAILED
            ],
            WorkflowState.LITERATURE_SEARCH: [
                WorkflowState.SEMANTIC_ANALYSIS,
                WorkflowState.CROSS_MODAL_ANALYSIS,
                WorkflowState.FAILED
            ],
            WorkflowState.SEMANTIC_ANALYSIS: [
                WorkflowState.CROSS_MODAL_ANALYSIS,
                WorkflowState.FAILED
            ],
            WorkflowState.CROSS_MODAL_ANALYSIS: [
                WorkflowState.INSIGHT_GENERATION,
                WorkflowState.FAILED
            ],
            WorkflowState.INSIGHT_GENERATION: [
                WorkflowState.FINALIZING,
                WorkflowState.FAILED
            ],
            WorkflowState.FINALIZING: [
                WorkflowState.COMPLETED,
                WorkflowState.FAILED
            ],
            WorkflowState.COMPLETED: [],
            WorkflowState.FAILED: [],
            WorkflowState.CANCELLED: []
        }
    
    async def execute_workflow(self, query: ResearchQuery) -> ResearchResult:
        """
        Execute a complete research workflow.
        
        Args:
            query: Research query to process
            
        Returns:
            Complete research results
        """
        # Create workflow context
        context = WorkflowContext(query=query)
        self.active_workflows[context.workflow_id] = context
        
        try:
            self.logger.info(f"Starting workflow {context.workflow_id} for query: {query.query_text}")
            
            # Execute workflow steps in sequence
            await self._execute_workflow_sequence(context)
            
            # Build final result
            result = self._build_research_result(context)
            
            self.logger.info(
                f"Workflow {context.workflow_id} completed in {context.duration:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Workflow {context.workflow_id} failed: {e}")
            context.current_state = WorkflowState.FAILED
            context.errors.append(str(e))
            
            # Return partial results even on failure
            return self._build_research_result(context)
            
        finally:
            context.end_time = datetime.utcnow()
            # Keep workflow in memory for debugging/monitoring
    
    async def _execute_workflow_sequence(self, context: WorkflowContext):
        """Execute workflow steps in proper sequence."""
        # Define execution order (can be parallelized later)
        execution_sequence = [
            WorkflowState.INITIALIZING,
            WorkflowState.CHEMICAL_ANALYSIS,
            WorkflowState.LITERATURE_SEARCH,
            WorkflowState.SEMANTIC_ANALYSIS,
            WorkflowState.CROSS_MODAL_ANALYSIS,
            WorkflowState.INSIGHT_GENERATION,
            WorkflowState.FINALIZING
        ]
        
        for state in execution_sequence:
            if context.is_complete:
                break
                
            # Check if step should be executed based on query parameters
            if not self._should_execute_step(state, context):
                self.logger.info(f"Skipping step {state.value} based on query parameters")
                continue
            
            try:
                await self._execute_step(state, context)
            except Exception as e:
                self.logger.error(f"Step {state.value} failed: {e}")
                
                # Determine if failure is fatal
                step = self.workflow_steps[state]
                if step.required:
                    context.current_state = WorkflowState.FAILED
                    context.errors.append(f"Required step {state.value} failed: {e}")
                    break
                else:
                    context.warnings.append(f"Optional step {state.value} failed: {e}")
                    continue
        
        # Mark as completed if we reached the end successfully
        if not context.is_complete and context.current_state != WorkflowState.FAILED:
            context.current_state = WorkflowState.COMPLETED
    
    def _should_execute_step(self, state: WorkflowState, context: WorkflowContext) -> bool:
        """Determine if a step should be executed based on query parameters."""
        query = context.query
        
        # Step execution logic based on query settings
        if state == WorkflowState.CHEMICAL_ANALYSIS:
            return query.include_chemical_similarity and query.compound_smiles
        
        if state == WorkflowState.LITERATURE_SEARCH:
            return query.include_literature_mining
        
        if state == WorkflowState.SEMANTIC_ANALYSIS:
            return query.include_literature_mining and len(context.literature_papers) > 0
        
        if state == WorkflowState.CROSS_MODAL_ANALYSIS:
            return (query.include_chemical_similarity and query.include_literature_mining and
                    len(context.chemical_compounds) > 0 and len(context.literature_papers) > 0)
        
        # Default: execute step
        return True
    
    async def _execute_step(self, state: WorkflowState, context: WorkflowContext):
        """Execute a single workflow step with error handling and retries."""
        step = self.workflow_steps[state]
        
        # Check dependencies
        for dependency in step.dependencies:
            if dependency not in [s.state for s in self.workflow_steps.values() 
                                 if s.state in context.step_results]:
                raise RuntimeError(f"Dependency {dependency.value} not satisfied for {state.value}")
        
        # Transition to new state
        self._transition_state(context, state)
        
        # Execute step with retries
        for attempt in range(step.retry_count):
            try:
                start_time = datetime.utcnow()
                
                # Execute step handler with timeout
                result = await asyncio.wait_for(
                    step.handler(context),
                    timeout=step.timeout_seconds
                )
                
                # Record timing
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                context.step_timings[step.name] = execution_time
                context.step_results[state.value] = result
                
                self.logger.info(
                    f"Step {step.name} completed in {execution_time:.2f}s (attempt {attempt + 1})"
                )
                return result
                
            except asyncio.TimeoutError:
                error_msg = f"Step {step.name} timed out after {step.timeout_seconds}s"
                self.logger.warning(f"{error_msg} (attempt {attempt + 1})")
                
                if attempt == step.retry_count - 1:
                    raise RuntimeError(error_msg)
                
                await asyncio.sleep(step.retry_delay_seconds)
                
            except Exception as e:
                error_msg = f"Step {step.name} failed: {e}"
                self.logger.warning(f"{error_msg} (attempt {attempt + 1})")
                
                if attempt == step.retry_count - 1:
                    raise e
                
                await asyncio.sleep(step.retry_delay_seconds)
    
    def _transition_state(self, context: WorkflowContext, new_state: WorkflowState):
        """Transition workflow to new state with validation."""
        current_state = context.current_state
        valid_transitions = self.state_transitions.get(current_state, [])
        
        if new_state not in valid_transitions and current_state != WorkflowState.PENDING:
            raise RuntimeError(
                f"Invalid state transition from {current_state.value} to {new_state.value}"
            )
        
        context.previous_state = current_state
        context.current_state = new_state
        
        self.logger.debug(f"Workflow {context.workflow_id}: {current_state.value} -> {new_state.value}")
    
    # Step Implementation Methods
    
    async def _initialize_workflow(self, context: WorkflowContext) -> Dict[str, Any]:
        """Initialize workflow and validate inputs."""
        query = context.query
        
        # Validate research query
        if not query.query_text or len(query.query_text.strip()) == 0:
            raise ValueError("Research query text is required")
        
        # Validate chemical inputs if provided
        if query.compound_smiles:
            is_valid, error_msg = self.chemical_service.validate_smiles(query.compound_smiles)
            if not is_valid:
                raise ValueError(f"Invalid SMILES: {error_msg}")
        
        # Set up workflow parameters
        initialization_data = {
            "query_text": query.query_text,
            "compound_name": query.compound_name,
            "compound_smiles": query.compound_smiles,
            "max_papers": query.max_papers,
            "include_chemical_similarity": query.include_chemical_similarity,
            "include_literature_mining": query.include_literature_mining,
            "estimated_duration": self._estimate_workflow_duration(query)
        }
        
        context.insights.append(f"Workflow initialized for query: '{query.query_text}'")
        
        return initialization_data
    
    async def _execute_chemical_analysis(self, context: WorkflowContext) -> Dict[str, Any]:
        """Execute chemical analysis workflow step."""
        query = context.query
        
        if not query.compound_smiles:
            return {"message": "No compound SMILES provided for analysis"}
        
        # Analyze query compound
        query_compound = ChemicalCompound(
            id="query_compound",
            name=query.compound_name or "Query Compound",
            smiles=query.compound_smiles
        )
        
        # Enrich with calculated properties
        enriched_compound = self.chemical_service.enrich_compound_data(query_compound)
        context.chemical_compounds.append(enriched_compound)
        
        # Find similar compounds if requested
        similar_compounds = []
        if query.include_chemical_similarity:
            # Use demo compound database for similarity search
            demo_compounds = self._get_demo_compound_database()
            
            similar_compounds = self.chemical_service.find_similar_compounds(
                query_smiles=query.compound_smiles,
                compound_database=demo_compounds,
                similarity_threshold=0.3,
                max_results=10
            )
            context.similar_compounds.extend(similar_compounds)
        
        # Generate chemical insights
        if enriched_compound.drug_likeness:
            if enriched_compound.drug_likeness > 0.7:
                context.insights.append("Query compound shows high drug-likeness potential")
            elif enriched_compound.drug_likeness < 0.3:
                context.insights.append("Query compound may have drug-likeness challenges")
        
        if similar_compounds:
            context.insights.append(f"Found {len(similar_compounds)} structurally similar compounds")
        
        return {
            "analyzed_compounds": len(context.chemical_compounds),
            "similar_compounds": len(similar_compounds),
            "drug_likeness": enriched_compound.drug_likeness
        }
    
    async def _execute_literature_search(self, context: WorkflowContext) -> Dict[str, Any]:
        """Execute literature search workflow step."""
        query = context.query
        
        # Primary literature search
        papers = await self.literature_service.search_pubmed(
            query=query.query_text,
            max_results=query.max_papers,
            date_from=query.date_from,
            date_to=query.date_to
        )
        
        context.literature_papers.extend(papers)
        
        # Compound-specific search if available
        if query.compound_name:
            compound_papers = await self.literature_service.search_pubmed(
                query=f"{query.compound_name} drug pharmacology",
                max_results=min(5, query.max_papers // 2),
                date_from=query.date_from,
                date_to=query.date_to
            )
            
            # Merge results, avoiding duplicates
            existing_pmids = {p.pubmed_id for p in context.literature_papers if p.pubmed_id}
            new_papers = [p for p in compound_papers if p.pubmed_id not in existing_pmids]
            context.literature_papers.extend(new_papers)
        
        # Generate literature insights
        if papers:
            context.insights.append(f"Retrieved {len(papers)} relevant research papers")
            
            # Analyze publication trends
            recent_papers = [
                p for p in papers 
                if p.publication_date and 
                (datetime.utcnow() - p.publication_date).days <= 365
            ]
            
            if recent_papers:
                context.insights.append(
                    f"{len(recent_papers)} papers published in the last year "
                    f"({len(recent_papers)/len(papers)*100:.1f}% of total)"
                )
        else:
            context.warnings.append("No literature papers found for the query")
        
        return {
            "papers_found": len(context.literature_papers),
            "recent_papers": len([p for p in context.literature_papers 
                                if p.publication_date and 
                                (datetime.utcnow() - p.publication_date).days <= 365])
        }
    
    async def _execute_semantic_analysis(self, context: WorkflowContext) -> Dict[str, Any]:
        """Execute semantic analysis workflow step."""
        query = context.query
        
        if not context.literature_papers:
            return {"message": "No literature papers available for semantic analysis"}
        
        # Find semantically similar papers
        similar_papers = await self.embedding_service.find_similar_papers(
            query_text=query.query_text,
            papers=context.literature_papers,
            similarity_threshold=0.5,
            max_results=5
        )
        
        context.semantic_results.extend(similar_papers)
        
        # Generate semantic insights
        if similar_papers:
            avg_similarity = sum(r.similarity_score for r in similar_papers) / len(similar_papers)
            context.insights.append(
                f"Average semantic similarity to query: {avg_similarity:.2f}"
            )
            
            # Identify most relevant paper
            top_paper = similar_papers[0]
            paper_title = top_paper.target_data.get('title', 'Unknown')
            context.insights.append(
                f"Most relevant paper: '{paper_title[:50]}...' "
                f"(similarity: {top_paper.similarity_score:.2f})"
            )
        
        return {
            "similar_papers": len(similar_papers),
            "avg_similarity": sum(r.similarity_score for r in similar_papers) / len(similar_papers) if similar_papers else 0
        }
    
    async def _execute_cross_modal_analysis(self, context: WorkflowContext) -> Dict[str, Any]:
        """Execute cross-modal analysis combining chemical and literature data."""
        
        # Map compounds to relevant literature
        compound_paper_mapping = {}
        
        for compound in context.chemical_compounds:
            relevant_papers = []
            
            for paper in context.literature_papers:
                # Check if compound name appears in paper
                title_text = paper.title.lower()
                abstract_text = paper.abstract.lower() if paper.abstract else ""
                full_text = title_text + " " + abstract_text
                
                if compound.name.lower() in full_text:
                    relevant_papers.append(paper)
            
            if relevant_papers:
                compound_paper_mapping[compound.name] = len(relevant_papers)
        
        # Generate cross-modal insights
        if compound_paper_mapping:
            most_studied = max(compound_paper_mapping.items(), key=lambda x: x[1])
            context.insights.append(
                f"Most studied compound: {most_studied[0]} "
                f"({most_studied[1]} papers)"
            )
        
        # Analyze research gaps
        research_gaps = []
        if len(context.literature_papers) < 5:
            research_gaps.append("Limited literature coverage for this research area")
        
        if len(context.chemical_compounds) > 0 and len(context.similar_compounds) == 0:
            research_gaps.append("No structurally similar compounds identified")
        
        return {
            "compound_paper_mappings": len(compound_paper_mapping),
            "research_gaps": research_gaps
        }
    
    async def _generate_insights(self, context: WorkflowContext) -> Dict[str, Any]:
        """Generate final research insights and recommendations."""
        
        # Calculate confidence score
        confidence_factors = []
        
        # Literature coverage factor
        if len(context.literature_papers) >= 10:
            confidence_factors.append(0.8)
        elif len(context.literature_papers) >= 5:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
        
        # Chemical analysis factor
        if context.chemical_compounds:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.4)
        
        # Semantic analysis factor
        if context.semantic_results:
            avg_sim = sum(r.similarity_score for r in context.semantic_results) / len(context.semantic_results)
            confidence_factors.append(avg_sim)
        else:
            confidence_factors.append(0.5)
        
        context.confidence_score = sum(confidence_factors) / len(confidence_factors)
        
        # Generate final insights
        if context.confidence_score > 0.7:
            context.insights.append("High confidence in research results")
        elif context.confidence_score > 0.5:
            context.insights.append("Moderate confidence in research results")
        else:
            context.insights.append("Low confidence - consider expanding search criteria")
        
        return {
            "confidence_score": context.confidence_score,
            "total_insights": len(context.insights)
        }
    
    async def _finalize_workflow(self, context: WorkflowContext) -> Dict[str, Any]:
        """Finalize workflow and prepare results."""
        
        # Final validation
        if not context.literature_papers and not context.chemical_compounds:
            context.warnings.append("No significant results found")
        
        # Generate summary statistics
        summary = {
            "workflow_duration": context.duration or (datetime.utcnow() - context.start_time).total_seconds(),
            "chemical_compounds_analyzed": len(context.chemical_compounds),
            "literature_papers_found": len(context.literature_papers),
            "similar_compounds_found": len(context.similar_compounds),
            "semantic_matches": len(context.semantic_results),
            "insights_generated": len(context.insights),
            "warnings": len(context.warnings),
            "errors": len(context.errors)
        }
        
        self.logger.info(f"Workflow {context.workflow_id} finalized: {summary}")
        
        return summary
    
    def _build_research_result(self, context: WorkflowContext) -> ResearchResult:
        """Build final research result from workflow context."""
        
        # Determine final status
        if context.current_state == WorkflowState.COMPLETED:
            status = ResearchStatus.COMPLETED
        elif context.current_state == WorkflowState.FAILED:
            status = ResearchStatus.FAILED
        else:
            status = ResearchStatus.IN_PROGRESS
        
        return ResearchResult(
            query_id=context.workflow_id,
            status=status,
            chemical_compounds=context.chemical_compounds,
            literature_papers=context.literature_papers,
            similar_compounds=context.similar_compounds,
            key_insights=context.insights,
            research_gaps=[],  # Could be extracted from cross-modal analysis
            processing_time_seconds=context.duration,
            confidence_score=context.confidence_score,
            created_at=context.start_time,
            completed_at=context.end_time,
            errors=context.errors,
            warnings=context.warnings
        )
    
    def _estimate_workflow_duration(self, query: ResearchQuery) -> float:
        """Estimate workflow duration based on query parameters."""
        base_time = 5.0  # Base workflow overhead
        
        if query.include_chemical_similarity:
            base_time += 3.0
        
        if query.include_literature_mining:
            base_time += query.max_papers * 0.1  # Rough estimate
        
        return min(base_time, 60.0)  # Cap at 1 minute estimate
    
    def _get_demo_compound_database(self) -> List[ChemicalCompound]:
        """Get demo compound database for similarity search."""
        return [
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
    
    # Workflow Management Methods
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow."""
        context = self.active_workflows.get(workflow_id)
        if not context:
            return None
        
        return {
            "workflow_id": workflow_id,
            "current_state": context.current_state.value,
            "progress": self._calculate_progress(context),
            "duration": context.duration or (datetime.utcnow() - context.start_time).total_seconds(),
            "step_timings": context.step_timings,
            "insights_count": len(context.insights),
            "errors_count": len(context.errors),
            "warnings_count": len(context.warnings)
        }
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active workflow."""
        context = self.active_workflows.get(workflow_id)
        if not context or context.is_complete:
            return False
        
        context.current_state = WorkflowState.CANCELLED
        context.end_time = datetime.utcnow()
        context.warnings.append("Workflow was cancelled by user")
        
        self.logger.info(f"Workflow {workflow_id} cancelled")
        return True
    
    def _calculate_progress(self, context: WorkflowContext) -> float:
        """Calculate workflow progress as percentage."""
        total_steps = len(self.workflow_steps)
        completed_steps = len(context.step_results)
        
        if context.current_state == WorkflowState.COMPLETED:
            return 100.0
        elif context.current_state == WorkflowState.FAILED:
            return min(completed_steps / total_steps * 100, 95.0)
        else:
            return completed_steps / total_steps * 100
    
    def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get overall workflow engine metrics."""
        total_workflows = len(self.active_workflows)
        completed_workflows = sum(
            1 for ctx in self.active_workflows.values() 
            if ctx.current_state == WorkflowState.COMPLETED
        )
        failed_workflows = sum(
            1 for ctx in self.active_workflows.values() 
            if ctx.current_state == WorkflowState.FAILED
        )
        
        return {
            "total_workflows": total_workflows,
            "completed_workflows": completed_workflows,
            "failed_workflows": failed_workflows,
            "success_rate": completed_workflows / total_workflows * 100 if total_workflows > 0 else 0,
            "active_workflows": total_workflows - completed_workflows - failed_workflows
        }