"""
Chemical compound processing service using RDKit.
Handles molecular similarity, property calculation, and drug-likeness analysis.
"""
import logging
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
    from rdkit.Chem.Fingerprints import FingerprintMols
    from rdkit.DataStructs import TanimotoSimilarity
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available. Chemical processing will be limited.")

from ..models import ChemicalCompound, CompoundType, SimilarityResult


@dataclass
class MolecularProperties:
    """Molecular property calculation results."""
    molecular_weight: float
    logp: float
    hbd: int  # Hydrogen bond donors
    hba: int  # Hydrogen bond acceptors
    tpsa: float  # Topological polar surface area
    rotatable_bonds: int
    aromatic_rings: int
    qed_score: float  # Quantitative Estimate of Drug-likeness


class ChemicalService:
    """Service for chemical compound processing and analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._validate_dependencies()
        
        # Drug-likeness thresholds (Lipinski's Rule of Five + extensions)
        self.lipinski_thresholds = {
            'molecular_weight': 500,
            'logp': 5,
            'hbd': 5,
            'hba': 10,
            'tpsa': 140,
            'rotatable_bonds': 10
        }
    
    def _validate_dependencies(self):
        """Validate that required chemical processing libraries are available."""
        if not RDKIT_AVAILABLE:
            self.logger.error("RDKit is required for chemical processing")
            raise ImportError("RDKit not available. Install with: pip install rdkit")
    
    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """
        Convert SMILES string to RDKit molecule object.
        
        Args:
            smiles: SMILES notation string
            
        Returns:
            RDKit molecule object or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self.logger.warning(f"Invalid SMILES: {smiles}")
                return None
            return mol
        except Exception as e:
            self.logger.error(f"Error parsing SMILES {smiles}: {e}")
            return None
    
    def calculate_molecular_properties(self, smiles: str) -> Optional[MolecularProperties]:
        """
        Calculate molecular properties from SMILES.
        
        Args:
            smiles: SMILES notation string
            
        Returns:
            MolecularProperties object or None if calculation fails
        """
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return None
        
        try:
            properties = MolecularProperties(
                molecular_weight=Descriptors.MolWt(mol),
                logp=Crippen.MolLogP(mol),
                hbd=Descriptors.NumHDonors(mol),
                hba=Descriptors.NumHAcceptors(mol),
                tpsa=Descriptors.TPSA(mol),
                rotatable_bonds=Descriptors.NumRotatableBonds(mol),
                aromatic_rings=Descriptors.NumAromaticRings(mol),
                qed_score=QED.qed(mol)
            )
            
            self.logger.debug(f"Calculated properties for {smiles}: MW={properties.molecular_weight:.2f}")
            return properties
            
        except Exception as e:
            self.logger.error(f"Error calculating properties for {smiles}: {e}")
            return None
    
    def assess_drug_likeness(self, properties: MolecularProperties) -> Tuple[float, List[str]]:
        """
        Assess drug-likeness based on molecular properties.
        
        Args:
            properties: Calculated molecular properties
            
        Returns:
            Tuple of (drug_likeness_score, violations_list)
        """
        violations = []
        score_components = []
        
        # Lipinski's Rule of Five violations
        if properties.molecular_weight > self.lipinski_thresholds['molecular_weight']:
            violations.append(f"MW > {self.lipinski_thresholds['molecular_weight']}")
            score_components.append(0.0)
        else:
            score_components.append(1.0)
            
        if properties.logp > self.lipinski_thresholds['logp']:
            violations.append(f"LogP > {self.lipinski_thresholds['logp']}")
            score_components.append(0.0)
        else:
            score_components.append(1.0)
            
        if properties.hbd > self.lipinski_thresholds['hbd']:
            violations.append(f"HBD > {self.lipinski_thresholds['hbd']}")
            score_components.append(0.0)
        else:
            score_components.append(1.0)
            
        if properties.hba > self.lipinski_thresholds['hba']:
            violations.append(f"HBA > {self.lipinski_thresholds['hba']}")
            score_components.append(0.0)
        else:
            score_components.append(1.0)
        
        # Additional drug-likeness factors
        if properties.tpsa > self.lipinski_thresholds['tpsa']:
            violations.append(f"TPSA > {self.lipinski_thresholds['tpsa']}")
            score_components.append(0.0)
        else:
            score_components.append(1.0)
            
        if properties.rotatable_bonds > self.lipinski_thresholds['rotatable_bonds']:
            violations.append(f"Rotatable bonds > {self.lipinski_thresholds['rotatable_bonds']}")
            score_components.append(0.0)
        else:
            score_components.append(1.0)
        
        # Combine QED score with rule-based assessment
        rule_based_score = sum(score_components) / len(score_components)
        combined_score = (rule_based_score * 0.7) + (properties.qed_score * 0.3)
        
        return combined_score, violations
    
    def calculate_similarity(self, smiles1: str, smiles2: str) -> Optional[float]:
        """
        Calculate Tanimoto similarity between two compounds.
        
        Args:
            smiles1: First compound SMILES
            smiles2: Second compound SMILES
            
        Returns:
            Similarity score (0-1) or None if calculation fails
        """
        mol1 = self.smiles_to_mol(smiles1)
        mol2 = self.smiles_to_mol(smiles2)
        
        if mol1 is None or mol2 is None:
            return None
        
        try:
            fp1 = FingerprintMols.FingerprintMol(mol1)
            fp2 = FingerprintMols.FingerprintMol(mol2)
            similarity = TanimotoSimilarity(fp1, fp2)
            
            self.logger.debug(f"Similarity between compounds: {similarity:.3f}")
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return None
    
    def find_similar_compounds(
        self, 
        query_smiles: str, 
        compound_database: List[ChemicalCompound],
        similarity_threshold: float = 0.7,
        max_results: int = 10
    ) -> List[SimilarityResult]:
        """
        Find similar compounds from a database.
        
        Args:
            query_smiles: Query compound SMILES
            compound_database: List of compounds to search
            similarity_threshold: Minimum similarity score
            max_results: Maximum number of results
            
        Returns:
            List of similarity results sorted by score
        """
        results = []
        
        for compound in compound_database:
            if not compound.smiles:
                continue
                
            similarity = self.calculate_similarity(query_smiles, compound.smiles)
            if similarity is None or similarity < similarity_threshold:
                continue
            
            result = SimilarityResult(
                target_id=compound.id or compound.name,
                similarity_score=similarity,
                similarity_type="tanimoto_fingerprint",
                target_data={
                    "name": compound.name,
                    "smiles": compound.smiles,
                    "molecular_weight": compound.molecular_weight,
                    "compound_type": compound.compound_type.value
                }
            )
            results.append(result)
        
        # Sort by similarity score (descending) and limit results
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:max_results]
    
    def enrich_compound_data(self, compound: ChemicalCompound) -> ChemicalCompound:
        """
        Enrich compound data with calculated properties.
        
        Args:
            compound: Input compound with basic data
            
        Returns:
            Enriched compound with calculated properties
        """
        if not compound.smiles:
            self.logger.warning(f"No SMILES available for compound: {compound.name}")
            return compound
        
        # Calculate molecular properties
        properties = self.calculate_molecular_properties(compound.smiles)
        if properties is None:
            return compound
        
        # Update compound with calculated properties
        compound.molecular_weight = properties.molecular_weight
        
        # Calculate drug-likeness and bioactivity estimates
        drug_likeness_score, violations = self.assess_drug_likeness(properties)
        compound.drug_likeness = drug_likeness_score
        
        # Simple bioactivity estimate (placeholder - in real system would use ML model)
        bioactivity_estimate = self._estimate_bioactivity(properties)
        compound.bioactivity_score = bioactivity_estimate
        
        # Simple toxicity estimate (placeholder)
        toxicity_estimate = self._estimate_toxicity(properties)
        compound.toxicity_score = toxicity_estimate
        
        # Determine compound type based on molecular weight
        if properties.molecular_weight < 900:
            compound.compound_type = CompoundType.SMALL_MOLECULE
        else:
            compound.compound_type = CompoundType.PROTEIN
        
        self.logger.info(
            f"Enriched compound {compound.name}: "
            f"MW={properties.molecular_weight:.1f}, "
            f"Drug-likeness={drug_likeness_score:.2f}"
        )
        
        return compound
    
    def _estimate_bioactivity(self, properties: MolecularProperties) -> float:
        """
        Simple bioactivity estimation based on molecular properties.
        In production, this would use a trained ML model.
        """
        # Simple heuristic based on drug-like properties
        score = 0.5  # Base score
        
        # Favor moderate molecular weight
        if 150 <= properties.molecular_weight <= 500:
            score += 0.2
        
        # Favor moderate lipophilicity
        if 0 <= properties.logp <= 3:
            score += 0.2
        
        # Consider QED score
        score += properties.qed_score * 0.3
        
        return min(1.0, max(0.0, score))
    
    def _estimate_toxicity(self, properties: MolecularProperties) -> float:
        """
        Simple toxicity estimation based on molecular properties.
        In production, this would use a trained ML model.
        """
        # Simple heuristic - higher values indicate higher toxicity risk
        risk_score = 0.1  # Base risk
        
        # High molecular weight increases risk
        if properties.molecular_weight > 600:
            risk_score += 0.3
        
        # Very high lipophilicity increases risk
        if properties.logp > 5:
            risk_score += 0.4
        
        # Many rotatable bonds may indicate instability
        if properties.rotatable_bonds > 8:
            risk_score += 0.2
        
        return min(1.0, max(0.0, risk_score))
    
    def validate_smiles(self, smiles: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SMILES notation.
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not smiles or not isinstance(smiles, str):
            return False, "SMILES must be a non-empty string"
        
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return False, "Invalid SMILES notation"
        
        return True, None