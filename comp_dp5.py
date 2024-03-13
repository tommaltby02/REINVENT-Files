"""DP5"""

from __future__ import annotations

__all__ = ["DP5"]

import os
import tempfile
import pickle
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
    python_path: List[str]
    pydp4_path: List[str]
    workflow: List[str]
    nmr_file: List[str]


@add_tag("__component")
class DP5:

    workflow_dict = {'dp5': 'w', 'cmae': 's', 'cmax': 's', 'rmse':'s'}

    def __init__(self, params: Parameters) -> np.array:
        self.python_path = params.python_path[0]
        self.pydp4_path = params.pydp4_path[0]
        self.workflow = params.workflow[0].lower()
        self.nmr_file = params.nmr_file[0]

    def __call__(self, smilies: List[str]) -> Any:
        scores = []
        cwd = os.getcwd()

        with tempfile.TemporaryDirectory() as temp_dir:
            input_smi_file = os.path.join(temp_dir, 'input.smiles')

            logger.info("Starting DP5 calculations in %s" % temp_dir)
            bad_ids, good_smi = self._prepare_input_data(smilies, input_smi_file)

            logger.info("Following molecules did not embed: %s" % str(bad_ids))

        # create temporary folder
            os.chdir(temp_dir)
            command = [self.python_path ,self.pydp4_path,
                            '--Smiles', input_smi_file, self.nmr_file,
                            "-w", self.workflow_dict[self.workflow],
                            "--OutputFolder", temp_dir]
            
            logger.info("Running the command...")
            logger.debug(' '.join(command))
            result = run_command(command)
            os.chdir(cwd)

            raw_scores = self._parse_output_data(temp_dir, good_smi)
            
            bad_no = 0
            if self.workflow == "dp5":
                bad_no = 0
            else:
                bad_no = 100
            
            

            for id in sorted(bad_ids):
                raw_scores.insert(id, bad_no)

            scores.append(np.array(raw_scores))

            return ComponentResults(scores)
            


    def _prepare_input_data(self, smiles: List[str], path: str):
        bad_ids = []
        good_smi = 0
        with open(path, 'w') as f:
            for i, smi in enumerate(smiles):
                mol = Chem.MolFromSmiles(Chem.CanonSmiles(smi))
                mol_h = AllChem.AddHs(mol, addCoords=True)
                cid = AllChem.EmbedMolecule(mol_h, forceTol=0.001, randomSeed=42)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_h)
                    f.write(f"{smi}\n")
                    good_smi += 1
                except ValueError:
                    bad_ids.append(i)
        return bad_ids, good_smi
    

    def _parse_output_data(self, path: str, good_smi: int) -> Tuple[List[str], List[float]]:
        # load the data_dic thing which contains the max score

        logger.debug("reading files at %s"% path)
        if self.workflow == 'dp5':
            dp5_path = f"{path}/dp5/data_dic.p"

            if not os.path.isfile(dp5_path):
                raise FileNotFoundError(f"Output file {path} has not finished with DP5 calculation.")

            with open(dp5_path, 'rb') as f:
                data = pickle.load(f)                
            # data is now a dictionary
            # get DP5_Exp_probs
            raw_scores = data['DP5_Exp_probs']

        elif self.workflow == 'cmae' or self.workflow == 'cmax' or self.workflow == 'rmse':
            dp4_path = f"{path}/dp4/data_dic.p"

            if not os.path.isfile(dp4_path):
                raw_scores = [100 for i in range(good_smi)]
            else:
                with open(dp4_path, 'rb') as f:
                    data = pickle.load(f) 

                    c_errors = data['Cerrors']
                    raw_scores = []
            

                    for isomer in c_errors:
                        errors = np.array(isomer)
                        if self.workflow == 'cmae':
                            result = np.abs(errors).mean()
                        elif self.workflow == 'cmax':
                            result = np.max(errors)
                        elif self.workflow == 'rmse':
                            result = np.sqrt(np.sum(errors**2) / len(errors))
                        raw_scores.append(result)

        return raw_scores

