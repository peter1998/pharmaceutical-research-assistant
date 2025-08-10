"""
Demo data utilities for pharmaceutical research assistant.
Provides sample compounds, literature, and research scenarios for demonstrations.
"""
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..models import ChemicalCompound, LiteraturePaper, ResearchQuery, CompoundType


logger = logging.getLogger(__name__)


class PharmaceuticalDemoData:
    """Demo data provider for pharmaceutical research scenarios."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Known pharmaceutical compounds with SMILES
        self._sample_compounds = {
            "aspirin": {
                "name": "Aspirin",
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "description": "Non-steroidal anti-inflammatory drug (NSAID)",
                "therapeutic_class": "Analgesic, Anti-inflammatory",
                "molecular_weight": 180.16,
                "drug_likeness": 0.85,
                "bioactivity_score": 0.92,
                "toxicity_score": 0.15
            },
            "ibuprofen": {
                "name": "Ibuprofen", 
                "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                "description": "Non-steroidal anti-inflammatory drug",
                "therapeutic_class": "NSAID",
                "molecular_weight": 206.28,
                "drug_likeness": 0.78,
                "bioactivity_score": 0.88,
                "toxicity_score": 0.12
            },
            "caffeine": {
                "name": "Caffeine",
                "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "description": "Central nervous system stimulant",
                "therapeutic_class": "Stimulant",
                "molecular_weight": 194.19,
                "drug_likeness": 0.72,
                "bioactivity_score": 0.76,
                "toxicity_score": 0.25
            },
            "morphine": {
                "name": "Morphine",
                "smiles": "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",
                "description": "Powerful opioid analgesic",
                "therapeutic_class": "Opioid Analgesic",
                "molecular_weight": 285.34,
                "drug_likeness": 0.45,
                "bioactivity_score": 0.95,
                "toxicity_score": 0.75
            },
            "penicillin_g": {
                "name": "Penicillin G",
                "smiles": "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C",
                "description": "Beta-lactam antibiotic",
                "therapeutic_class": "Antibiotic",
                "molecular_weight": 334.39,
                "drug_likeness": 0.65,
                "bioactivity_score": 0.89,
                "toxicity_score": 0.18
            },
            "acetaminophen": {
                "name": "Acetaminophen",
                "smiles": "CC(=O)NC1=CC=C(C=C1)O",
                "description": "Analgesic and antipyretic",
                "therapeutic_class": "Analgesic",
                "molecular_weight": 151.16,
                "drug_likeness": 0.81,
                "bioactivity_score": 0.74,
                "toxicity_score": 0.35
            },
            "warfarin": {
                "name": "Warfarin",
                "smiles": "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",
                "description": "Anticoagulant medication",
                "therapeutic_class": "Anticoagulant",
                "molecular_weight": 308.33,
                "drug_likeness": 0.58,
                "bioactivity_score": 0.82,
                "toxicity_score": 0.68
            },
            "insulin_lispro": {
                "name": "Insulin Lispro",
                "smiles": "CC[C@H](C)[C@H](NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC1=CC=C(C=C1)O)NC(=O)[C@H](CC(=O)N)NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CCCCN)NC(=O)[C@H](CC2=CC=CC=C2)NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](C)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CC(=O)N)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCC(=O)N)NC(=O)[C@H](C(C)O)NC(=O)[C@H](CC3=CC=C(C=C3)O)NC(=O)[C@H](CC4=CNC5=CC=CC=C54)NC(=O)[C@H](CC(=O)N)NC(=O)[C@H](CO)NC(=O)[C@H](CC6=CC=CC=C6)NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](C)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CC(=O)N)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCC(=O)N)NC(=O)[C@H](C(C)O)NC(=O)[C@H](CC7=CC=C(C=C7)O)NC(=O)[C@H](CC8=CNC9=CC=CC=C98)NC(=O)[C@H](CC(=O)N)NC(=O)[C@H](CO)NC(=O)[C@H](CC1=CC=CC=C1)NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](C)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CC(=O)N)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCC(=O)N)NC(=O)[C@H](C(C)O)NC(=O)[C@H](CC1=CC=C(C=C1)O)NC(=O)[C@H](CC1=CNC2=CC=CC=C21)NC(=O)[C@H](CC(=O)N)NC(=O)[C@H](CO)NC(=O)CNC(=O)[C@H](CCC(=O)N)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC3=CC=CC=C3)NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](C)N)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CC(=O)N)C(=O)O",
                "description": "Fast-acting insulin analog",
                "therapeutic_class": "Insulin",
                "molecular_weight": 5808.0,
                "drug_likeness": 0.15,
                "bioactivity_score": 0.98,
                "toxicity_score": 0.08
            }
        }
        
        # Sample research scenarios
        self._research_scenarios = [
            {
                "name": "Anti-inflammatory Drug Discovery",
                "description": "Research novel anti-inflammatory compounds",
                "query_text": "anti-inflammatory drugs mechanism COX inhibition prostaglandin",
                "compound_focus": "ibuprofen",
                "expected_insights": [
                    "COX-2 selective inhibitors show promise",
                    "Prostaglandin pathway modulation",
                    "Reduced gastrointestinal side effects"
                ]
            },
            {
                "name": "Opioid Alternative Research",
                "description": "Find non-addictive pain management alternatives",
                "query_text": "pain management non-opioid analgesics addiction potential",
                "compound_focus": "morphine",
                "expected_insights": [
                    "NMDA receptor antagonists show promise",
                    "Cannabinoid pathways for pain relief",
                    "Nerve growth factor inhibitors"
                ]
            },
            {
                "name": "Antibiotic Resistance Study",
                "description": "Research mechanisms of antibiotic resistance",
                "query_text": "antibiotic resistance beta lactam mechanisms novel antibiotics",
                "compound_focus": "penicillin_g",
                "expected_insights": [
                    "Beta-lactamase inhibitor combinations",
                    "Novel bacterial targets identified",
                    "Combination therapy approaches"
                ]
            }
        ]
    
    def get_sample_compounds(self) -> List[ChemicalCompound]:
        """
        Get list of sample pharmaceutical compounds.
        
        Returns:
            List of ChemicalCompound objects
        """
        compounds = []
        
        for compound_id, data in self._sample_compounds.items():
            try:
                compound = ChemicalCompound(
                    id=compound_id,
                    name=data["name"],
                    smiles=data["smiles"],
                    molecular_weight=data.get("molecular_weight"),
                    drug_likeness=data.get("drug_likeness"),
                    bioactivity_score=data.get("bioactivity_score"),
                    toxicity_score=data.get("toxicity_score"),
                    compound_type=self._determine_compound_type(data.get("molecular_weight", 0)),
                    source="demo_data"
                )
                compounds.append(compound)
                
            except Exception as e:
                self.logger.warning(f"Error creating compound {compound_id}: {e}")
                continue
        
        return compounds
    
    def get_compound_by_name(self, name: str) -> Optional[ChemicalCompound]:
        """
        Get specific compound by name.
        
        Args:
            name: Compound name
            
        Returns:
            ChemicalCompound object or None
        """
        name_lower = name.lower()
        
        for compound_id, data in self._sample_compounds.items():
            if data["name"].lower() == name_lower or compound_id == name_lower:
                return ChemicalCompound(
                    id=compound_id,
                    name=data["name"],
                    smiles=data["smiles"],
                    molecular_weight=data.get("molecular_weight"),
                    drug_likeness=data.get("drug_likeness"),
                    bioactivity_score=data.get("bioactivity_score"),
                    toxicity_score=data.get("toxicity_score"),
                    compound_type=self._determine_compound_type(data.get("molecular_weight", 0)),
                    source="demo_data"
                )
        
        return None
    
    def get_sample_literature(self) -> List[LiteraturePaper]:
        """
        Get sample literature papers for demonstration.
        
        Returns:
            List of LiteraturePaper objects
        """
        papers = [
            LiteraturePaper(
                pubmed_id="12345678",
                doi="10.1038/nature12345",
                title="Novel COX-2 Selective Inhibitors for Anti-inflammatory Therapy",
                abstract="This study investigates the development of selective cyclooxygenase-2 (COX-2) inhibitors that demonstrate potent anti-inflammatory activity with reduced gastrointestinal toxicity. We synthesized a series of compounds and evaluated their selectivity profiles.",
                authors=["Smith, J.A.", "Johnson, M.B.", "Williams, R.C."],
                journal="Nature",
                publication_date=datetime.now() - timedelta(days=90),
                keywords=["COX-2", "anti-inflammatory", "selectivity", "toxicity"],
                chemical_entities=["ibuprofen", "celecoxib", "rofecoxib"],
                relevance_score=0.92,
                source="demo_data"
            ),
            LiteraturePaper(
                pubmed_id="23456789",
                doi="10.1056/NEJMoa234567",
                title="Comparative Efficacy of Non-Opioid Analgesics in Chronic Pain Management",
                abstract="A comprehensive meta-analysis of non-opioid analgesics for chronic pain management, evaluating efficacy, safety profiles, and addiction potential. Results suggest promising alternatives to traditional opioid therapy.",
                authors=["Brown, A.L.", "Davis, K.M.", "Wilson, P.J.", "Taylor, S.R."],
                journal="New England Journal of Medicine",
                publication_date=datetime.now() - timedelta(days=45),
                keywords=["chronic pain", "non-opioid", "analgesics", "addiction"],
                chemical_entities=["gabapentin", "pregabalin", "duloxetine"],
                relevance_score=0.88,
                source="demo_data"
            ),
            LiteraturePaper(
                pubmed_id="34567890",
                doi="10.1016/j.cmi.2023.01.001",
                title="Mechanisms of Beta-Lactam Resistance and Novel Antibiotic Strategies",
                abstract="Investigation of emerging beta-lactam resistance mechanisms in clinical isolates and evaluation of novel antibiotic combinations to overcome resistance. Focus on beta-lactamase inhibitor development.",
                authors=["Garcia, M.A.", "Lee, S.H.", "Patel, R.K."],
                journal="Clinical Microbiology and Infection",
                publication_date=datetime.now() - timedelta(days=120),
                keywords=["beta-lactam", "resistance", "antibiotics", "beta-lactamase"],
                chemical_entities=["penicillin", "clavulanic acid", "tazobactam"],
                relevance_score=0.85,
                source="demo_data"
            ),
            LiteraturePaper(
                pubmed_id="45678901",
                doi="10.1124/jpet.123.000456",
                title="Pharmacokinetic Properties of Modified-Release Analgesic Formulations",
                abstract="Comparative pharmacokinetic study of immediate-release versus modified-release formulations of common analgesics. Analysis of absorption, distribution, metabolism, and excretion profiles.",
                authors=["Liu, X.Y.", "Anderson, T.M.", "Rodriguez, C.A."],
                journal="Journal of Pharmacology and Experimental Therapeutics",
                publication_date=datetime.now() - timedelta(days=60),
                keywords=["pharmacokinetics", "modified-release", "analgesics", "ADME"],
                chemical_entities=["acetaminophen", "ibuprofen", "tramadol"],
                relevance_score=0.79,
                source="demo_data"
            ),
            LiteraturePaper(
                pubmed_id="56789012",
                doi="10.1021/acs.jmedchem.3c00123",
                title="Structure-Activity Relationships in Novel Anti-Cancer Compounds",
                abstract="Systematic analysis of structure-activity relationships in a series of novel anti-cancer compounds derived from natural products. Molecular docking and cytotoxicity studies reveal promising leads.",
                authors=["Zhang, Q.L.", "Thompson, K.R.", "Martinez, J.F.", "Chen, W.X."],
                journal="Journal of Medicinal Chemistry",
                publication_date=datetime.now() - timedelta(days=30),
                keywords=["anti-cancer", "structure-activity", "natural products", "cytotoxicity"],
                chemical_entities=["paclitaxel", "doxorubicin", "cisplatin"],
                relevance_score=0.82,
                source="demo_data"
            )
        ]
        
        return papers
    
    def get_research_scenarios(self) -> List[Dict[str, Any]]:
        """
        Get predefined research scenarios for demonstration.
        
        Returns:
            List of research scenario dictionaries
        """
        return self._research_scenarios.copy()
    
    def create_demo_research_query(self, scenario_name: str) -> Optional[ResearchQuery]:
        """
        Create a research query for a specific scenario.
        
        Args:
            scenario_name: Name of the research scenario
            
        Returns:
            ResearchQuery object or None
        """
        scenario = next(
            (s for s in self._research_scenarios if s["name"] == scenario_name),
            None
        )
        
        if not scenario:
            return None
        
        compound_data = self._sample_compounds.get(scenario["compound_focus"])
        
        return ResearchQuery(
            query_text=scenario["query_text"],
            compound_smiles=compound_data["smiles"] if compound_data else None,
            compound_name=compound_data["name"] if compound_data else None,
            max_papers=15,
            include_chemical_similarity=True,
            include_literature_mining=True
        )
    
    def get_similarity_test_pairs(self) -> List[Dict[str, Any]]:
        """
        Get compound pairs for similarity testing.
        
        Returns:
            List of compound pair dictionaries with expected similarities
        """
        return [
            {
                "name": "NSAID Similarity",
                "compound1": {"name": "Aspirin", "smiles": self._sample_compounds["aspirin"]["smiles"]},
                "compound2": {"name": "Ibuprofen", "smiles": self._sample_compounds["ibuprofen"]["smiles"]},
                "expected_similarity": 0.45,
                "description": "Both are NSAIDs with similar mechanisms"
            },
            {
                "name": "Different Classes",
                "compound1": {"name": "Aspirin", "smiles": self._sample_compounds["aspirin"]["smiles"]},
                "compound2": {"name": "Caffeine", "smiles": self._sample_compounds["caffeine"]["smiles"]},
                "expected_similarity": 0.15,
                "description": "Different therapeutic classes"
            },
            {
                "name": "Analgesic Comparison",
                "compound1": {"name": "Morphine", "smiles": self._sample_compounds["morphine"]["smiles"]},
                "compound2": {"name": "Acetaminophen", "smiles": self._sample_compounds["acetaminophen"]["smiles"]},
                "expected_similarity": 0.25,
                "description": "Both analgesics but different mechanisms"
            }
        ]
    
    def generate_performance_test_data(self, num_compounds: int = 100) -> List[ChemicalCompound]:
        """
        Generate synthetic compound data for performance testing.
        
        Args:
            num_compounds: Number of compounds to generate
            
        Returns:
            List of synthetic ChemicalCompound objects
        """
        import random
        import string
        
        compounds = []
        base_smiles = [
            "CCO",  # Ethanol
            "CC(=O)O",  # Acetic acid
            "c1ccccc1",  # Benzene
            "CCN(CC)CC",  # Triethylamine
            "CC(C)C",  # Isopropyl
        ]
        
        for i in range(num_compounds):
            # Generate synthetic compound
            base = random.choice(base_smiles)
            name = f"TestCompound_{i:04d}"
            
            compound = ChemicalCompound(
                id=f"test_{i}",
                name=name,
                smiles=base,
                molecular_weight=random.uniform(100, 500),
                drug_likeness=random.uniform(0.1, 0.9),
                bioactivity_score=random.uniform(0.2, 0.95),
                toxicity_score=random.uniform(0.05, 0.8),
                compound_type=random.choice(list(CompoundType)),
                source="performance_test"
            )
            compounds.append(compound)
        
        return compounds
    
    def save_demo_data_json(self, filepath: str = "demo_data.json"):
        """
        Save demo data to JSON file for external use.
        
        Args:
            filepath: Path to save JSON file
        """
        try:
            demo_data = {
                "compounds": {},
                "literature": [],
                "scenarios": self._research_scenarios,
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "version": "1.0.0",
                    "description": "Pharmaceutical research assistant demo data"
                }
            }
            
            # Convert compounds
            for compound_id, data in self._sample_compounds.items():
                demo_data["compounds"][compound_id] = data
            
            # Convert literature
            papers = self.get_sample_literature()
            for paper in papers:
                paper_dict = paper.dict()
                # Convert datetime to string
                if paper_dict.get("publication_date"):
                    paper_dict["publication_date"] = paper_dict["publication_date"].isoformat()
                if paper_dict.get("retrieved_at"):
                    paper_dict["retrieved_at"] = paper_dict["retrieved_at"].isoformat()
                demo_data["literature"].append(paper_dict)
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(demo_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Demo data saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving demo data: {e}")
            raise
    
    def _determine_compound_type(self, molecular_weight: float) -> CompoundType:
        """Determine compound type based on molecular weight."""
        if molecular_weight < 900:
            return CompoundType.SMALL_MOLECULE
        elif molecular_weight < 10000:
            return CompoundType.PEPTIDE
        else:
            return CompoundType.PROTEIN
    
    def get_demo_summary(self) -> Dict[str, Any]:
        """
        Get summary of available demo data.
        
        Returns:
            Summary statistics dictionary
        """
        compounds = self.get_sample_compounds()
        papers = self.get_sample_literature()
        
        return {
            "compounds": {
                "total": len(compounds),
                "by_type": {
                    ctype.value: len([c for c in compounds if c.compound_type == ctype])
                    for ctype in CompoundType
                },
                "avg_molecular_weight": sum(c.molecular_weight or 0 for c in compounds) / len(compounds),
                "avg_drug_likeness": sum(c.drug_likeness or 0 for c in compounds) / len(compounds)
            },
            "literature": {
                "total": len(papers),
                "date_range": {
                    "earliest": min(p.publication_date for p in papers if p.publication_date).isoformat(),
                    "latest": max(p.publication_date for p in papers if p.publication_date).isoformat()
                },
                "avg_relevance": sum(p.relevance_score or 0 for p in papers) / len(papers),
                "journals": list(set(p.journal for p in papers if p.journal))
            },
            "scenarios": {
                "total": len(self._research_scenarios),
                "names": [s["name"] for s in self._research_scenarios]
            }
        }


# Global demo data instance
demo_data = PharmaceuticalDemoData()


def get_demo_data() -> PharmaceuticalDemoData:
    """Get global demo data instance."""
    return demo_data


if __name__ == "__main__":
    # Command line interface for demo data
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "summary":
            summary = demo_data.get_demo_summary()
            print(json.dumps(summary, indent=2))
            
        elif command == "compounds":
            compounds = demo_data.get_sample_compounds()
            for compound in compounds:
                print(f"{compound.name}: {compound.smiles}")
                
        elif command == "literature":
            papers = demo_data.get_sample_literature()
            for paper in papers:
                print(f"{paper.title} ({paper.journal})")
                
        elif command == "save":
            output_file = sys.argv[2] if len(sys.argv) > 2 else "demo_data.json"
            demo_data.save_demo_data_json(output_file)
            print(f"Demo data saved to {output_file}")
            
        else:
            print("Available commands: summary, compounds, literature, save")
    else:
        print("Pharmaceutical Research Assistant - Demo Data")
        print("Usage: python demo_data.py [command]")
        print("Commands: summary, compounds, literature, save")