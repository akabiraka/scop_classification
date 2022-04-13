import sys
sys.path.append("../scop_classification")

from Bio.PDB import *
from Bio.PDB.PDBIO import Select

class CAAtomSelector(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id
    
    def  accept_chain(self, chain):
        # print(chain.id)
        if chain.id == self.chain_id:
            return 1
        else:
            return 0
        
    def accept_atom(self, atom):
        """Overload this to reject atoms for output."""
        if atom.name == "CA":
            return 1
        else:
            return 0        
        
class StandardAminoAcidSelector(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id
        
    def  accept_chain(self, chain):
        # print(chain.id, self.chain_id)
        if self.chain_id == chain.id:
            return 1
        else:
            return 0
            
    def accept_residue(self, residue):
        if residue.get_resname() in standard_aa_names:
            return 1
        else:
            return 0

class AllBackboneAtomSelector(Select):
    """Backbone atoms: CA, CB, N, O

    Args:
        Select ([type]): [description]
    """
    def __init__(self, chain_id):
        self.chain_id = chain_id
    
    def  accept_chain(self, chain):
        # print(chain.id)
        if chain.id == self.chain_id:
            return 1
        else:
            return 0
        
    def accept_atom(self, atom):
        """Overload this to reject atoms for output."""
        if atom.name == "CA":
            return 1
        elif atom.name == "CB":
            return 1
        elif atom.name == "N":
            return 1
        elif atom.name == "O":
            return 1
        else:
            return 0
        
class ChainAndAminoAcidSelect(Select):
    def __init__(self, chain_id):
        super(ChainAndAminoAcidSelect).__init__()
        self.chain_id = chain_id
        
    def  accept_chain(self, chain):
        # print(chain.id, self.chain_id)
        if chain.id == self.chain_id:
            return 1
        else:
            return 0
            
    def accept_residue(self, residue):
        hetero_flag, sequence_identifier, insertion_code = residue.id
        if residue.get_resname() in standard_aa_names:# and insertion_code==" ":
            # print(residue.get_resname())
            return 1
        else:
            return 0

class ChainAndRegionSelect(Select):
    def __init__(self, chain_id, region) -> None:
        """
        Args:
            chain_id (char): "A"
            region (str): range in string inclusive, format example "2-113"
        """
        super(ChainAndRegionSelect, self).__init__()
        self.chain_id = chain_id
        if region.count("-")==1: 
            start, end = region.split("-")
        elif region.count("-")==2: 
            loc = region.index("-", region.index("-")+1)
            start, end = region[:loc], region[loc+1:]
        else: raise

        self.region_range = range(int(start), int(end)+1)

    def  accept_chain(self, chain):
        # print(chain.id, self.chain_id)
        if chain.id == self.chain_id:
            return 1
        else:
            return 0
    
    def accept_residue(self, residue):
        hetero_flag, sequence_identifier, insertion_code = residue.id
        if residue.get_resname() in standard_aa_names and sequence_identifier in self.region_range:
            # print(residue.get_resname())
            return 1
        else:
            return 0
