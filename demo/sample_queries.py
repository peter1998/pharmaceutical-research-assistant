"""
Sample queries for pharmaceutical research assistant demonstration.
These examples showcase different capabilities of the system.
"""
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Sample chemical compounds with SMILES for demonstration
SAMPLE_COMPOUNDS = {
    "aspirin": {
        "name": "Aspirin",
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "description": "Common pain reliever and anti-inflammatory drug"
    },
    "ibuprofen": {
        "name": "Ibuprofen", 
        "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "description": "Non-steroidal anti-inflammatory drug (NSAID)"
    },
    "caffeine": {
        "name": "Caffeine",
        "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "description": "Central nervous system stimulant"
    },
    "morphine": {
        "name": "Morphine",
        "smiles": "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",
        "description": "Powerful opioid pain medication"
    },
    "penicillin_g": {
        "name": "Penicillin G",
        "smiles": "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C",
        "description": "Beta-lactam antibiotic"
    }
}

# Sample research queries for different use cases
SAMPLE_RESEARCH_QUERIES = [
    {
        "name": "Drug Discovery - Anti-inflammatory",
        "query": {
            "query_text": "anti-inflammatory drugs mechanism of action COX inhibition",
            "compound_smiles": SAMPLE_COMPOUNDS["ibuprofen"]["smiles"],
            "compound_name": "Ibuprofen",
            "max_papers": 15,
            "include_chemical_similarity": True,
            "include_literature_mining": True
        },
        "description": "Research anti-inflammatory mechanisms and find similar compounds",
        "expected_results": "Literature on COX inhibition, similar NSAIDs, safety profiles"
    },
    
    {
        "name": "Pharmacokinetics Study",
        "query": {
            "query_text": "drug metabolism pharmacokinetics absorption distribution",
            "compound_smiles": SAMPLE_COMPOUNDS["caffeine"]["smiles"],
            "compound_name": "Caffeine",
            "max_papers": 12,
            "include_chemical_similarity": True,
            "include_literature_mining": True
        },
        "description": "Study pharmacokinetic properties of stimulants",
        "expected_results": "ADME data, metabolic pathways, similar stimulants"
    },
    
    {
        "name": "Antibiotic Resistance",
        "query": {
            "query_text": "antibiotic resistance beta lactam penicillin mechanisms",
            "compound_smiles": SAMPLE_COMPOUNDS["penicillin_g"]["smiles"],
            "compound_name": "Penicillin G",
            "max_papers": 20,
            "include_chemical_similarity": True,
            "include_literature_mining": True,
            "date_from": datetime.now() - timedelta(days=1825)  # Last 5 years
        },
        "description": "Research antibiotic resistance mechanisms",
        "expected_results": "Resistance studies, beta-lactamase data, alternative antibiotics"
    },
    
    {
        "name": "Pain Management Research",
        "query": {
            "query_text": "opioid receptors pain management addiction potential",
            "compound_smiles": SAMPLE_COMPOUNDS["morphine"]["smiles"],
            "compound_name": "Morphine",
            "max_papers": 10,
            "include_chemical_similarity": True,
            "include_literature_mining": True
        },
        "description": "Study opioid mechanisms and addiction potential",
        "expected_results": "Receptor binding data, addiction studies, alternative analgesics"
    },
    
    {
        "name": "Central Nervous System",
        "query": {
            "query_text": "central nervous system stimulants adenosine receptor",
            "compound_smiles": SAMPLE_COMPOUNDS["caffeine"]["smiles"],
            "compound_name": "Caffeine",
            "max_papers": 8,
            "include_chemical_similarity": True,
            "include_literature_mining": True
        },
        "description": "Research CNS stimulant mechanisms",
        "expected_results": "Adenosine receptor data, CNS effects, similar stimulants"
    }
]

# Sample chemical similarity queries
CHEMICAL_SIMILARITY_QUERIES = [
    {
        "name": "NSAID Similarity Search",
        "query_smiles": SAMPLE_COMPOUNDS["aspirin"]["smiles"],
        "candidate_smiles": [
            SAMPLE_COMPOUNDS["ibuprofen"]["smiles"],
            SAMPLE_COMPOUNDS["caffeine"]["smiles"],
            "CC(=O)NC1=CC=C(C=C1)O",  # Acetaminophen
            "CC1=CC(=NO1)C2=CC=CC=C2C(=O)O",  # Isoxicam
            "CC1=C(C(=O)N(N1C)C2=CC=CC=C2)C(=O)O"  # Phenylbutazone
        ],
        "threshold": 0.6,
        "description": "Find compounds similar to aspirin"
    },
    
    {
        "name": "Stimulant Similarity Search",
        "query_smiles": SAMPLE_COMPOUNDS["caffeine"]["smiles"],
        "candidate_smiles": [
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine (self)
            "CNC(=O)N(C)C1=NC=NC2=C1N=CN2C",  # Theophylline
            "CN(C)CCC=C1C2=CC=CC=C2SC3=C1C=C(C=C3)Cl",  # Chlorpromazine
            "CCN1C=C(C(=O)C2=CC(=C(N=C21)N3CCNCC3)F)C(=O)O"  # Ciprofloxacin
        ],
        "threshold": 0.5,
        "description": "Find compounds similar to caffeine"
    }
]

# Sample literature search queries
LITERATURE_QUERIES = [
    {
        "name": "Recent COVID Drug Research",
        "query": "COVID-19 antiviral drugs treatment",
        "max_results": 15,
        "date_from": datetime.now() - timedelta(days=730),  # Last 2 years
        "description": "Recent research on COVID-19 treatments"
    },
    
    {
        "name": "Cancer Immunotherapy",
        "query": "cancer immunotherapy checkpoint inhibitors PD-1",
        "max_results": 12,
        "description": "Cancer immunotherapy research"
    },
    
    {
        "name": "Alzheimer's Drug Development",
        "query": "Alzheimer disease drug development amyloid tau",
        "max_results": 10,
        "date_from": datetime.now() - timedelta(days=1095),  # Last 3 years
        "description": "Recent Alzheimer's drug development"
    },
    
    {
        "name": "Diabetes Treatment",
        "query": "diabetes mellitus insulin therapy glucose control",
        "max_results": 8,
        "description": "Diabetes treatment research"
    }
]

# Sample semantic similarity queries
SEMANTIC_QUERIES = [
    {
        "name": "Drug Mechanism Similarity",
        "query_text": "enzyme inhibition competitive binding active site",
        "candidate_texts": [
            "The compound binds competitively to the enzyme active site",
            "Non-competitive inhibition occurs at allosteric sites",
            "Irreversible binding leads to permanent enzyme inactivation",
            "Substrate concentration affects competitive inhibition",
            "Allosteric regulation modulates enzyme activity",
            "Covalent modification of enzyme residues",
            "This paper discusses weather patterns in tropical regions"
        ],
        "threshold": 0.6,
        "description": "Find texts similar to enzyme inhibition concepts"
    },
    
    {
        "name": "Pharmacology Concepts",
        "query_text": "drug absorption distribution metabolism excretion ADME",
        "candidate_texts": [
            "Pharmacokinetic properties determine drug efficacy",
            "Hepatic metabolism affects drug clearance rates",
            "Renal excretion is the primary elimination pathway",
            "Bioavailability depends on absorption mechanisms",
            "Protein binding influences drug distribution",
            "Drug-drug interactions affect metabolism",
            "The study examined plant growth in arid conditions"
        ],
        "threshold": 0.5,
        "description": "Find texts related to pharmacokinetics"
    }
]

def get_demo_queries() -> Dict[str, List[Dict[str, Any]]]:
    """
    Get all demo queries organized by category.
    
    Returns:
        Dictionary with query categories and their examples
    """
    return {
        "research_queries": SAMPLE_RESEARCH_QUERIES,
        "chemical_similarity": CHEMICAL_SIMILARITY_QUERIES,
        "literature_search": LITERATURE_QUERIES,
        "semantic_similarity": SEMANTIC_QUERIES,
        "compounds": SAMPLE_COMPOUNDS
    }

def get_quick_demo_sequence() -> List[Dict[str, Any]]:
    """
    Get a sequence of queries for a quick 5-minute demo.
    
    Returns:
        List of demo steps with descriptions
    """
    return [
        {
            "step": 1,
            "title": "Chemical Analysis",
            "description": "Analyze aspirin and calculate drug-likeness",
            "endpoint": "/api/v1/chemical/analyze",
            "data": {
                "name": SAMPLE_COMPOUNDS["aspirin"]["name"],
                "smiles": SAMPLE_COMPOUNDS["aspirin"]["smiles"]
            },
            "demo_notes": "Shows molecular property calculation, drug-likeness scoring"
        },
        
        {
            "step": 2,
            "title": "Chemical Similarity",
            "description": "Find compounds similar to ibuprofen",
            "endpoint": "/api/v1/chemical/similarity",
            "data": {
                "query_smiles": SAMPLE_COMPOUNDS["ibuprofen"]["smiles"],
                "candidate_smiles": [
                    SAMPLE_COMPOUNDS["aspirin"]["smiles"],
                    SAMPLE_COMPOUNDS["caffeine"]["smiles"],
                    "CC(=O)NC1=CC=C(C=C1)O"  # Acetaminophen
                ],
                "threshold": 0.3
            },
            "demo_notes": "Demonstrates Tanimoto similarity with molecular fingerprints"
        },
        
        {
            "step": 3,
            "title": "Literature Search",
            "description": "Search for pain management research",
            "endpoint": "/api/v1/literature/search",
            "data": {
                "query": "pain management NSAIDs anti-inflammatory",
                "max_results": 10
            },
            "demo_notes": "Shows PubMed integration and relevance scoring"
        },
        
        {
            "step": 4,
            "title": "Semantic Similarity",
            "description": "Find semantically related research concepts",
            "endpoint": "/api/v1/semantic/similarity",
            "data": SEMANTIC_QUERIES[0],
            "demo_notes": "Demonstrates embedding-based semantic search"
        },
        
        {
            "step": 5,
            "title": "Comprehensive Research",
            "description": "Full research workflow for anti-inflammatory drugs",
            "endpoint": "/api/v1/research/comprehensive",
            "data": SAMPLE_RESEARCH_QUERIES[0]["query"],
            "demo_notes": "Shows complete multi-modal research pipeline"
        }
    ]

def print_demo_summary():
    """Print a summary of available demo queries."""
    queries = get_demo_queries()
    
    print("=== Pharmaceutical Research Assistant Demo Queries ===\n")
    
    print(f"📊 Sample Compounds: {len(queries['compounds'])}")
    for name, data in queries['compounds'].items():
        print(f"  • {data['name']}: {data['description']}")
    
    print(f"\n🔬 Research Queries: {len(queries['research_queries'])}")
    for i, query in enumerate(queries['research_queries'], 1):
        print(f"  {i}. {query['name']}: {query['description']}")
    
    print(f"\n🧪 Chemical Similarity: {len(queries['chemical_similarity'])}")
    for i, query in enumerate(queries['chemical_similarity'], 1):
        print(f"  {i}. {query['name']}: {query['description']}")
    
    print(f"\n📚 Literature Search: {len(queries['literature_search'])}")
    for i, query in enumerate(queries['literature_search'], 1):
        print(f"  {i}. {query['name']}: {query['description']}")
    
    print(f"\n🔍 Semantic Similarity: {len(queries['semantic_queries'])}")
    for i, query in enumerate(queries['semantic_similarity'], 1):
        print(f"  {i}. {query['name']}: {query['description']}")

def save_demo_data_json(filename: str = "demo_data.json"):
    """Save demo data to JSON file for easy loading."""
    demo_data = get_demo_queries()
    
    # Convert datetime objects to strings for JSON serialization
    for query in demo_data['research_queries']:
        if 'date_from' in query['query'] and query['query']['date_from']:
            query['query']['date_from'] = query['query']['date_from'].isoformat()
        if 'date_to' in query['query'] and query['query']['date_to']:
            query['query']['date_to'] = query['query']['date_to'].isoformat()
    
    for query in demo_data['literature_search']:
        if 'date_from' in query and query['date_from']:
            query['date_from'] = query['date_from'].isoformat()
        if 'date_to' in query and query['date_to']:
            query['date_to'] = query['date_to'].isoformat()
    
    with open(filename, 'w') as f:
        json.dump(demo_data, f, indent=2)
    
    print(f"Demo data saved to {filename}")

if __name__ == "__main__":
    print_demo_summary()
    print("\n" + "="*60)
    
    # Quick demo sequence
    demo_sequence = get_quick_demo_sequence()
    print(f"\n🎯 Quick Demo Sequence ({len(demo_sequence)} steps):")
    for step in demo_sequence:
        print(f"  Step {step['step']}: {step['title']}")
        print(f"    → {step['description']}")
        print(f"    → {step['demo_notes']}")
        print()
    
    # Save demo data
    save_demo_data_json()