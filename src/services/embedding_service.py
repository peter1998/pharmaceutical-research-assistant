"""
Text embedding service using local sentence-transformer models.
Handles vector generation, similarity search, and semantic text analysis.
"""
import logging
import asyncio
import hashlib
from typing import List, Optional, Dict, Tuple
import numpy as np
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Embedding features will be limited.")

from ..models import EmbeddingVector, LiteraturePaper, SimilarityResult
from ..config import settings


class EmbeddingService:
    """Service for text embeddings and semantic similarity."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_name = settings.embedding.model_name
        self.max_sequence_length = settings.embedding.max_sequence_length
        self.batch_size = settings.embedding.batch_size
        
        self._model = None
        self._embedding_cache = {}  # Simple in-memory cache
        
        self._validate_dependencies()
        self._load_model()
    
    def _validate_dependencies(self):
        """Validate that required embedding libraries are available."""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.error("sentence-transformers is required for embedding service")
            raise ImportError(
                "sentence-transformers not available. "
                "Install with: pip install sentence-transformers"
            )
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            
            # Set device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(device)
            
            self.logger.info(f"Model loaded successfully on device: {device}")
            
            # Test model with a simple sentence
            test_embedding = self._model.encode("test sentence")
            self.embedding_dimension = len(test_embedding)
            self.logger.info(f"Embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            self.logger.error(f"Error loading embedding model: {e}")
            raise RuntimeError(f"Failed to load embedding model: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for embedding generation.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Basic text cleaning
        text = text.strip()
        
        # Truncate if too long
        if len(text) > self.max_sequence_length * 4:  # Rough character estimate
            text = text[:self.max_sequence_length * 4]
            self.logger.debug(f"Truncated text to {len(text)} characters")
        
        return text
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector or None if generation fails
        """
        if not text:
            return None
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self._embedding_cache:
            self.logger.debug("Using cached embedding")
            return self._embedding_cache[cache_key]
        
        try:
            preprocessed_text = self._preprocess_text(text)
            if not preprocessed_text:
                return None
            
            # Generate embedding (run in thread to avoid blocking)
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, 
                self._model.encode, 
                preprocessed_text
            )
            
            # Convert to list and cache
            embedding_list = embedding.tolist()
            self._embedding_cache[cache_key] = embedding_list
            
            self.logger.debug(f"Generated embedding for text length: {len(text)}")
            return embedding_list
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            return None
    
    async def generate_batch_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors (None for failed generations)
        """
        if not texts:
            return []
        
        try:
            # Preprocess texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            valid_indices = [i for i, text in enumerate(processed_texts) if text]
            valid_texts = [processed_texts[i] for i in valid_indices]
            
            if not valid_texts:
                return [None] * len(texts)
            
            # Generate embeddings in batches
            all_embeddings = []
            for i in range(0, len(valid_texts), self.batch_size):
                batch_texts = valid_texts[i:i + self.batch_size]
                
                # Run in thread to avoid blocking
                loop = asyncio.get_event_loop()
                batch_embeddings = await loop.run_in_executor(
                    None, 
                    self._model.encode, 
                    batch_texts
                )
                
                all_embeddings.extend(batch_embeddings)
            
            # Map back to original indices
            result_embeddings = [None] * len(texts)
            for i, embedding in enumerate(all_embeddings):
                original_index = valid_indices[i]
                result_embeddings[original_index] = embedding.tolist()
                
                # Cache the embedding
                cache_key = self._get_cache_key(texts[original_index])
                self._embedding_cache[cache_key] = embedding.tolist()
            
            self.logger.info(f"Generated {len(all_embeddings)} embeddings from {len(texts)} texts")
            return result_embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating batch embeddings: {e}")
            return [None] * len(texts)
    
    def calculate_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> Optional[float]:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0-1) or None if calculation fails
        """
        try:
            if not embedding1 or not embedding2:
                return None
            
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Convert to 0-1 range (cosine similarity is -1 to 1)
            normalized_similarity = (similarity + 1) / 2
            
            return float(normalized_similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return None
    
    async def find_similar_texts(
        self, 
        query_text: str, 
        candidate_texts: List[str],
        similarity_threshold: float = 0.7,
        max_results: int = 10
    ) -> List[SimilarityResult]:
        """
        Find similar texts using semantic similarity.
        
        Args:
            query_text: Query text
            candidate_texts: List of candidate texts to compare
            similarity_threshold: Minimum similarity score
            max_results: Maximum number of results
            
        Returns:
            List of similarity results
        """
        try:
            # Generate embedding for query
            query_embedding = await self.generate_embedding(query_text)
            if not query_embedding:
                self.logger.warning("Failed to generate query embedding")
                return []
            
            # Generate embeddings for candidates
            candidate_embeddings = await self.generate_batch_embeddings(candidate_texts)
            
            # Calculate similarities
            results = []
            for i, candidate_embedding in enumerate(candidate_embeddings):
                if candidate_embedding is None:
                    continue
                
                similarity = self.calculate_similarity(query_embedding, candidate_embedding)
                if similarity is None or similarity < similarity_threshold:
                    continue
                
                result = SimilarityResult(
                    target_id=str(i),
                    similarity_score=similarity,
                    similarity_type="semantic_cosine",
                    target_data={
                        "text": candidate_texts[i][:200] + "..." if len(candidate_texts[i]) > 200 else candidate_texts[i],
                        "full_text_length": len(candidate_texts[i])
                    }
                )
                results.append(result)
            
            # Sort by similarity score and limit results
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results[:max_results]
            
        except Exception as e:
            self.logger.error(f"Error finding similar texts: {e}")
            return []
    
    async def find_similar_papers(
        self, 
        query_text: str, 
        papers: List[LiteraturePaper],
        similarity_threshold: float = 0.6,
        max_results: int = 10
    ) -> List[SimilarityResult]:
        """
        Find similar research papers using semantic similarity.
        
        Args:
            query_text: Query text
            papers: List of literature papers
            similarity_threshold: Minimum similarity score
            max_results: Maximum number of results
            
        Returns:
            List of similarity results with paper metadata
        """
        try:
            # Create text representations of papers
            paper_texts = []
            for paper in papers:
                # Combine title and abstract for semantic matching
                paper_text = paper.title
                if paper.abstract:
                    paper_text += " " + paper.abstract
                paper_texts.append(paper_text)
            
            # Find similar texts
            similarity_results = await self.find_similar_texts(
                query_text=query_text,
                candidate_texts=paper_texts,
                similarity_threshold=similarity_threshold,
                max_results=max_results
            )
            
            # Enhance results with paper metadata
            enhanced_results = []
            for result in similarity_results:
                paper_index = int(result.target_id)
                paper = papers[paper_index]
                
                enhanced_result = SimilarityResult(
                    target_id=paper.pubmed_id or f"paper_{paper_index}",
                    similarity_score=result.similarity_score,
                    similarity_type="semantic_paper",
                    target_data={
                        "title": paper.title,
                        "authors": paper.authors[:3] if paper.authors else [],  # First 3 authors
                        "journal": paper.journal,
                        "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
                        "pubmed_id": paper.pubmed_id,
                        "doi": paper.doi,
                        "keywords": paper.keywords[:5] if paper.keywords else [],  # First 5 keywords
                        "abstract_preview": paper.abstract[:300] + "..." if paper.abstract and len(paper.abstract) > 300 else paper.abstract
                    }
                )
                enhanced_results.append(enhanced_result)
            
            self.logger.info(f"Found {len(enhanced_results)} similar papers")
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Error finding similar papers: {e}")
            return []
    
    async def create_embedding_vector(self, text: str, vector_id: str) -> Optional[EmbeddingVector]:
        """
        Create an EmbeddingVector object with metadata.
        
        Args:
            text: Input text
            vector_id: Unique identifier for the vector
            
        Returns:
            EmbeddingVector object or None if creation fails
        """
        try:
            embedding = await self.generate_embedding(text)
            if not embedding:
                return None
            
            return EmbeddingVector(
                id=vector_id,
                text=text,
                vector=embedding,
                model_name=self.model_name,
                created_at=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Error creating embedding vector: {e}")
            return None
    
    def cluster_embeddings(
        self, 
        embeddings: List[List[float]], 
        n_clusters: int = 5
    ) -> List[int]:
        """
        Cluster embeddings using K-means.
        
        Args:
            embeddings: List of embedding vectors
            n_clusters: Number of clusters
            
        Returns:
            List of cluster labels
        """
        try:
            from sklearn.cluster import KMeans
            
            if len(embeddings) < n_clusters:
                return list(range(len(embeddings)))
            
            # Convert to numpy array
            embedding_matrix = np.array(embeddings)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embedding_matrix)
            
            self.logger.info(f"Clustered {len(embeddings)} embeddings into {n_clusters} clusters")
            return cluster_labels.tolist()
            
        except Exception as e:
            self.logger.error(f"Error clustering embeddings: {e}")
            return [0] * len(embeddings)  # Return all in one cluster as fallback
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "max_sequence_length": self.max_sequence_length,
            "device": str(self._model.device) if self._model else "unknown",
            "cache_size": len(self._embedding_cache),
            "is_loaded": self._model is not None
        }
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        self.logger.info("Embedding cache cleared")