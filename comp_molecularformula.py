"""molecularformula"""

from __future__ import annotations

__all__ = ["molecularformula"]

import os
import tempfile
import pickle
import guacamol 
from guacamol.common_scoring_functions import IsomerScoringFunction
from dataclasses import dataclass
from typing import Any, List, IO
import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .component_results import ComponentResults
from .run_program import run_command
from .add_tag import add_tag


logger = logging.getLogger('reinvent')

@add_tag("__parameters")
@dataclass
class Parameters:
    molecular_formula: List[str]
    mean_function: List[str]
 

@add_tag("__component")
class molecularformula:

    def __init__(self, params: Parameters) -> np.array:
        self.molecular_formula = params.molecular_formula[0]
        self.mean_function = params.mean_function[0]
    
    def __call__(self, smiles: List[str]) -> Any:
        scores = []
        raw_scores = []
        scorer = IsomerScoringFunction(self.molecular_formula, self.mean_function)
        for smile in smiles:
            score = scorer.raw_score(smile)
            raw_scores.append(score)
        scores.append(np.array(raw_scores))
        return ComponentResults(scores)
        
        
        
        
        
       
        
        
        
        
        
            


