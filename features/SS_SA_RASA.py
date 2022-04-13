import sys
sys.path.append("../scop_classification")
from Bio.PDB import PDBParser, DSSP
import numpy as np

class SS_SA_RASA(object):
    """ This uses DSSP module. See readme of how to install 
    """
    def __init__(self, ss_dict=None) -> None:
        super().__init__()
        # helix: H,G,I; sheet:B,E; coil:T,S,-
        self.ss_dict = ss_dict if ss_dict is not None else {"H":"H", "G":"H", "I":"H", "B":"B", "E":"B", "T":"C", "S":"C", "-":"C"}

    def get_ss_onehot(self, ss, smooth_encoding=True):
        letter = self.ss_dict.get(ss)
        if smooth_encoding: return np.array([0.1 if char != letter else 0.9 for char in "HBC"], dtype=np.float32)
        else: return np.array([0.0 if char != letter else 1.0 for char in "HBC"], dtype=np.float32)


    def get_sa_onehot(self, rasa, smooth_encoding=True):
        letter = self.get_sa_type(rasa)
        if smooth_encoding: return np.array([0.1 if char != letter else 0.9 for char in "BEI"], dtype=np.float32)
        else: return np.array([0.0 if char != letter else 1.0 for char in "BEI"], dtype=np.float32)


    def get_ss_sa_rasa(self, pdb_id, chain_id, cln_pdb_file, mutation_site):
        ss, rasa = self.get_ss_and_rasa_at_residue(pdb_id, chain_id, cln_pdb_file, mutation_site)
        ss_onehot, sa_onehot=self.get_ss_onehot(ss), self.get_sa_onehot(rasa)
        return ss_onehot, sa_onehot, rasa


    def get_ss_and_rasa_at_residue(self, cln_pdb_file, chain_id, residue_id):
        """ss:Secondary structure. rasa:Relative accessible surface area
        """
        model = PDBParser(QUIET=True).get_structure("", cln_pdb_file)[0]
        residue_id = (" ", residue_id[1], residue_id[2]) #removed the hetero-flag 
        dssp = DSSP(model, cln_pdb_file, dssp="mkdssp")

        if (chain_id, residue_id) in dssp.keys(): ss, rasa = dssp[chain_id, residue_id][2], dssp[chain_id, residue_id][3]
        else: ss, rasa = "-", 0.5 # rasa=0.5 means intermediate, neither exposed nor buried
        
        return self.get_ss_onehot(ss), rasa

    def get_sa_type(self, rasa):
        if rasa<0.25: sa="B" # Buried
        elif rasa>0.5: sa="E" # Exposed
        else: sa="I" # Intermediate 
        return sa   

    def get_full_ss_and_sa(self, cln_pdb_file, chain_id, format="onehot"):
        """sa: solvent accessibility, ss:Secondary structure
        """
        model = PDBParser(QUIET=True).get_structure("", cln_pdb_file)[0]
        dssp = DSSP(model, cln_pdb_file, dssp="mkdssp")
        
        ss_types, sa_types, rasa_values = [], [], []
        for residue in model.get_residues():
            if (chain_id, residue.id) in dssp.keys(): ss, rasa = dssp[chain_id, residue.id][2], dssp[chain_id, residue.id][3]
            else: ss, rasa = "-", 0.5 # rasa=0.5 means intermediate, neither exposed nor buried
            
            rasa_values.append(rasa)
            if format=="onehot":
                ss_types.append(self.get_ss_onehot(ss))
                sa_types.append(self.get_sa_onehot(rasa))
                
            else:
                ss_types.append(self.ss_dict.get(ss))
                sa_types.append(self.get_sa_type(rasa))    
        
        if format=="onehot": return np.array(ss_types), np.array(sa_types), rasa_values
        else: return "".join(ss_types), "".join(sa_types), rasa_values


# ss_sa_rasa = SS_SA_RASA()
# ss, sasa = ss_sa_rasa.get_ss_and_rasa_at_residue("data/pdbs_clean/3ecaA.pdb", "A", ('H_ASP', 401, ' '))
# print(ss, sasa)
# ss_types, sa_types, rasa_values = ss_sa_rasa.get_full_ss_and_sa("data/pdbs_clean/1a0fA.pdb", "A", format="onehot")
# print(ss_types)
# print(sa_types)
# print(rasa_values)
