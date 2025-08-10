"""
Literature search and analysis service using PubMed E-utilities.
Handles scientific paper retrieval, keyword extraction, and relevance scoring.
"""
import logging
import re
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Set
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import httpx
from ..models import LiteraturePaper
from ..config import settings


class LiteratureService:
    """Service for literature search and analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = settings.api.pubmed_base_url
        self.email = settings.api.pubmed_email
        self.api_key = settings.api.pubmed_api_key
        self.request_timeout = settings.api.request_timeout
        
        # Common pharmaceutical and chemical keywords for relevance scoring
        self.pharma_keywords = {
            'high_impact': [
                'drug discovery', 'pharmacokinetics', 'bioavailability', 'clinical trial',
                'mechanism of action', 'drug target', 'therapeutic', 'efficacy', 'safety',
                'toxicity', 'adverse effects', 'pharmacodynamics', 'metabolism'
            ],
            'medium_impact': [
                'molecular', 'compound', 'synthesis', 'activity', 'binding', 'receptor',
                'inhibitor', 'agonist', 'antagonist', 'bioactive', 'pharmaceutical'
            ],
            'chemical_entities': [
                'protein', 'enzyme', 'antibody', 'peptide', 'nucleic acid', 'small molecule',
                'organic compound', 'pharmaceutical compound'
            ]
        }
    
    async def search_pubmed(
        self, 
        query: str, 
        max_results: int = 20,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> List[LiteraturePaper]:
        """
        Search PubMed for scientific literature.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            date_from: Search from this date
            date_to: Search until this date
            
        Returns:
            List of literature papers
        """
        try:
            # Step 1: Search for PMIDs
            pmids = await self._search_pmids(query, max_results, date_from, date_to)
            if not pmids:
                self.logger.warning(f"No PMIDs found for query: {query}")
                return []
            
            # Step 2: Fetch detailed paper information
            papers = await self._fetch_paper_details(pmids)
            
            # Step 3: Calculate relevance scores
            for paper in papers:
                paper.relevance_score = self._calculate_relevance_score(paper, query)
            
            # Sort by relevance score
            papers.sort(key=lambda p: p.relevance_score or 0, reverse=True)
            
            self.logger.info(f"Retrieved {len(papers)} papers for query: {query}")
            return papers
            
        except Exception as e:
            self.logger.error(f"Error searching PubMed: {e}")
            return []
    
    async def _search_pmids(
        self, 
        query: str, 
        max_results: int,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> List[str]:
        """
        Search PubMed for PMIDs using E-search.
        
        Args:
            query: Search query
            max_results: Maximum results
            date_from: Start date filter
            date_to: End date filter
            
        Returns:
            List of PubMed IDs
        """
        # Build search URL
        search_url = f"{self.base_url}esearch.fcgi"
        
        # Prepare query parameters
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': min(max_results, 100),  # PubMed limit
            'retmode': 'xml',
            'tool': 'pharmaceutical_research_assistant',
            'email': self.email
        }
        
        # Add API key if available
        if self.api_key:
            params['api_key'] = self.api_key
        
        # Add date filters
        if date_from or date_to:
            date_filter = self._build_date_filter(date_from, date_to)
            params['term'] = f"({query}) AND {date_filter}"
        
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            try:
                response = await client.get(search_url, params=params)
                response.raise_for_status()
                
                # Parse XML response
                root = ET.fromstring(response.content)
                pmids = [id_elem.text for id_elem in root.findall('.//Id')]
                
                self.logger.debug(f"Found {len(pmids)} PMIDs for query: {query}")
                return pmids
                
            except httpx.HTTPError as e:
                self.logger.error(f"HTTP error searching PubMed: {e}")
                return []
            except ET.ParseError as e:
                self.logger.error(f"XML parsing error: {e}")
                return []
    
    async def _fetch_paper_details(self, pmids: List[str]) -> List[LiteraturePaper]:
        """
        Fetch detailed paper information using E-fetch.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of literature papers with details
        """
        if not pmids:
            return []
        
        # PubMed allows fetching multiple papers in one request
        fetch_url = f"{self.base_url}efetch.fcgi"
        
        params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml',
            'tool': 'pharmaceutical_research_assistant',
            'email': self.email
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            try:
                response = await client.get(fetch_url, params=params)
                response.raise_for_status()
                
                # Parse XML response
                root = ET.fromstring(response.content)
                papers = self._parse_pubmed_xml(root)
                
                self.logger.debug(f"Fetched details for {len(papers)} papers")
                return papers
                
            except httpx.HTTPError as e:
                self.logger.error(f"HTTP error fetching paper details: {e}")
                return []
            except ET.ParseError as e:
                self.logger.error(f"XML parsing error in paper details: {e}")
                return []
    
    def _parse_pubmed_xml(self, root: ET.Element) -> List[LiteraturePaper]:
        """
        Parse PubMed XML response into LiteraturePaper objects.
        
        Args:
            root: XML root element
            
        Returns:
            List of parsed papers
        """
        papers = []
        
        for article in root.findall('.//PubmedArticle'):
            try:
                paper = self._parse_single_article(article)
                if paper:
                    papers.append(paper)
            except Exception as e:
                self.logger.warning(f"Error parsing article: {e}")
                continue
        
        return papers
    
    def _parse_single_article(self, article: ET.Element) -> Optional[LiteraturePaper]:
        """
        Parse a single PubMed article XML element.
        
        Args:
            article: Article XML element
            
        Returns:
            LiteraturePaper object or None
        """
        try:
            # Extract PMID
            pmid_elem = article.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else None
            
            # Extract basic information
            title_elem = article.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else "No title"
            
            abstract_elem = article.find('.//AbstractText')
            abstract = abstract_elem.text if abstract_elem is not None else None
            
            # Extract authors
            authors = []
            for author in article.findall('.//Author'):
                last_name = author.find('LastName')
                first_name = author.find('ForeName')
                if last_name is not None:
                    author_name = last_name.text
                    if first_name is not None:
                        author_name += f", {first_name.text}"
                    authors.append(author_name)
            
            # Extract journal
            journal_elem = article.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else None
            
            # Extract publication date
            pub_date = self._extract_publication_date(article)
            
            # Extract DOI
            doi = self._extract_doi(article)
            
            # Extract keywords
            keywords = self._extract_keywords(title, abstract)
            
            # Extract chemical entities
            chemical_entities = self._extract_chemical_entities(title, abstract)
            
            paper = LiteraturePaper(
                pubmed_id=pmid,
                doi=doi,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                publication_date=pub_date,
                keywords=keywords,
                chemical_entities=chemical_entities,
                source="pubmed"
            )
            
            return paper
            
        except Exception as e:
            self.logger.error(f"Error parsing article: {e}")
            return None
    
    def _extract_publication_date(self, article: ET.Element) -> Optional[datetime]:
        """Extract publication date from article XML."""
        try:
            # Try different date elements
            for date_path in ['.//PubDate', './/ArticleDate']:
                date_elem = article.find(date_path)
                if date_elem is not None:
                    year_elem = date_elem.find('Year')
                    month_elem = date_elem.find('Month')
                    day_elem = date_elem.find('Day')
                    
                    if year_elem is not None:
                        year = int(year_elem.text)
                        month = int(month_elem.text) if month_elem is not None else 1
                        day = int(day_elem.text) if day_elem is not None else 1
                        
                        return datetime(year, month, day)
            return None
        except (ValueError, AttributeError):
            return None
    
    def _extract_doi(self, article: ET.Element) -> Optional[str]:
        """Extract DOI from article XML."""
        for article_id in article.findall('.//ArticleId'):
            if article_id.get('IdType') == 'doi':
                return article_id.text
        return None
    
    def _extract_keywords(self, title: str, abstract: Optional[str]) -> List[str]:
        """
        Extract relevant keywords from title and abstract.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            
        Returns:
            List of extracted keywords
        """
        text = title.lower()
        if abstract:
            text += " " + abstract.lower()
        
        keywords = set()
        
        # Check for pharmaceutical keywords
        for category, keyword_list in self.pharma_keywords.items():
            for keyword in keyword_list:
                if keyword.lower() in text:
                    keywords.add(keyword)
        
        # Extract compound names (simple pattern matching)
        compound_patterns = [
            r'\b[A-Z][a-z]+\s*\d+\b',  # Pattern like "Compound 123"
            r'\b[A-Z]{2,}\d+\b',       # Pattern like "ABC123"
        ]
        
        for pattern in compound_patterns:
            matches = re.findall(pattern, title + " " + (abstract or ""))
            keywords.update(matches)
        
        return list(keywords)[:10]  # Limit to 10 keywords
    
    def _extract_chemical_entities(self, title: str, abstract: Optional[str]) -> List[str]:
        """
        Extract chemical entity mentions from text.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            
        Returns:
            List of chemical entities
        """
        text = title.lower()
        if abstract:
            text += " " + abstract.lower()
        
        entities = set()
        
        # Check for chemical entity keywords
        for entity in self.pharma_keywords['chemical_entities']:
            if entity.lower() in text:
                entities.add(entity)
        
        return list(entities)[:5]  # Limit to 5 entities
    
    def _calculate_relevance_score(self, paper: LiteraturePaper, query: str) -> float:
        """
        Calculate relevance score for a paper based on query.
        
        Args:
            paper: Literature paper
            query: Original search query
            
        Returns:
            Relevance score (0-1)
        """
        score = 0.0
        query_lower = query.lower()
        
        # Title relevance (weight: 0.4)
        title_lower = paper.title.lower()
        title_score = self._text_similarity(query_lower, title_lower)
        score += title_score * 0.4
        
        # Abstract relevance (weight: 0.3)
        if paper.abstract:
            abstract_lower = paper.abstract.lower()
            abstract_score = self._text_similarity(query_lower, abstract_lower)
            score += abstract_score * 0.3
        
        # Keyword relevance (weight: 0.2)
        keyword_score = 0.0
        for keyword in paper.keywords:
            if keyword.lower() in query_lower:
                keyword_score += 0.1
        score += min(keyword_score, 0.2)
        
        # Recency boost (weight: 0.1)
        if paper.publication_date:
            days_old = (datetime.now() - paper.publication_date).days
            recency_score = max(0, 1 - (days_old / 3650))  # 10-year decay
            score += recency_score * 0.1
        
        return min(1.0, score)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity based on common words.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # Simple word overlap similarity
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _build_date_filter(
        self, 
        date_from: Optional[datetime], 
        date_to: Optional[datetime]
    ) -> str:
        """
        Build PubMed date filter string.
        
        Args:
            date_from: Start date
            date_to: End date
            
        Returns:
            Date filter string for PubMed query
        """
        if date_from and date_to:
            return f'("{date_from.strftime("%Y/%m/%d")}"[Date] : "{date_to.strftime("%Y/%m/%d")}"[Date])'
        elif date_from:
            return f'"{date_from.strftime("%Y/%m/%d")}"[Date] : "3000"[Date]'
        elif date_to:
            return f'"1900"[Date] : "{date_to.strftime("%Y/%m/%d")}"[Date]'
        else:
            return ""
    
    async def get_trending_topics(self, days: int = 30) -> List[str]:
        """
        Get trending topics in pharmaceutical research.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of trending topics
        """
        try:
            # Search for recent papers in pharmaceutical research
            date_from = datetime.now() - timedelta(days=days)
            
            trending_queries = [
                "drug discovery",
                "clinical trial",
                "biomarker",
                "personalized medicine",
                "immunotherapy"
            ]
            
            trending_topics = []
            
            for query in trending_queries:
                papers = await self.search_pubmed(
                    query=query,
                    max_results=10,
                    date_from=date_from
                )
                
                if papers:
                    # Extract common keywords from recent papers
                    all_keywords = []
                    for paper in papers:
                        all_keywords.extend(paper.keywords)
                    
                    # Count keyword frequency
                    keyword_counts = {}
                    for keyword in all_keywords:
                        keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
                    
                    # Get top keywords
                    top_keywords = sorted(
                        keyword_counts.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:3]
                    
                    trending_topics.extend([kw[0] for kw in top_keywords])
            
            return list(set(trending_topics))[:10]
            
        except Exception as e:
            self.logger.error(f"Error getting trending topics: {e}")
            return []