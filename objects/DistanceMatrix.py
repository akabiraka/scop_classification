import sys
sys.path.append("../scop_classification")
import numpy as np
import traceback
from Bio.PDB import PDBParser

class DistanceMatrix(object):
    def __init__(self):
        super(DistanceMatrix, self).__init__()
        self.bb_atoms = ['CA', 'CB', 'N', 'O']

    def compute_atom_atom_distance(self, residue_1, residue_2, atom_1="CB", atom_2="CB"):
        """Compute euclidean distance between two atoms of two residues.

        Args:
            residue_1 (Bio.PDB.Residue): 
            residue_2 (Bio.PDB.Residue): 
            atom_1 (str, optional): Defaults to "CB".
            atom_2 (str, optional): Defaults to "CB".

        Returns:
            float: euclidean distance
        """
        try:
            if atom_1=="CB" and residue_1.get_resname()=='GLY':
                atom_1 = "CA"
            if atom_2=="CB" and residue_2.get_resname()=='GLY':
                atom_2 = "CA"
            diff_vector = residue_1[atom_1].coord - residue_2[atom_2].coord
        except Exception as e:
            print("Can not resolve distance: ", residue_1.id, residue_1.get_resname(), residue_2.id, residue_2.get_resname(), atom_1, atom_2)
            # traceback.print_exc()
            # raise
            # in case, there is an error but I want the distance matrix, comment out above 2 lines and comment in next line
            return 0.0 
        return np.sqrt(np.sum(diff_vector * diff_vector))
                

    def compute_4n4n_distance_matrix(self, residue_list_1, residue_list_2):
        """Returns distance matrix.

        Args:
            residue_list_1 (list): 
            residue_list_2 (list): 

        Returns:
            numpy.matrix: [4N, 4N] where N=len(residue_list_1).
        """
        l = len(self.bb_atoms)
        dist_matrix = np.zeros((l*len(residue_list_1), l*len(residue_list_2)), float)
        for row, residue_1 in enumerate(residue_list_1):
            for col, residue_2 in enumerate(residue_list_2):
                for k, atom_1 in enumerate(self.bb_atoms):
                    for l, atom_2 in enumerate(self.bb_atoms):
                        dist_matrix[4*row+k, 4*col+l] = self.compute_atom_atom_distance(residue_1, residue_2, atom_1, atom_2)
        return dist_matrix  
    

    def compute_nn_distance_matrix(self, residue_list_1, residue_list_2, atom_1="CB", atom_2="CB"):
        """Returns distance matrix.

        Args:
            residue_list_1 (list): 
            residue_list_2 (list): 
            atom_1 (str, optional): Defaults to "CB".
            atom_2 (str, optional): Defaults to "CB".

        Returns:
            numpy.matrix: [N, N] where N=len(residue_list_1).
        """
        dist_matrix = np.zeros((len(residue_list_1), len(residue_list_2)), float)
        for row, residue_1 in enumerate(residue_list_1):
            for col, residue_2 in enumerate(residue_list_2):
                dist_matrix[row, col] = self.compute_atom_atom_distance(residue_1, residue_2, atom_1, atom_2)
        return dist_matrix 


    def get(self, cln_pdb_file, chain_id, matrix_type="NN", atom_1="CB", atom_2="CB"):
        """Returns distance matrix among all residues.

        Args:
            cln_pdb_file (str): filepath
            chain_id (str): i.e "A"
            matrix_type (str, optional): NN or 4N4N. Defaults to "NN".
            atom_1 (str, optional): Defaults to "CB".
            atom_2 (str, optional): Defaults to "CB".

        Returns:
            numpy.matrix: [N,N] or [4N, 4N] where N is the number of residues.
        """
        pdb_id = cln_pdb_file.split("/")[-1].split(".")[0]
        structure = PDBParser(QUIET=True).get_structure("", cln_pdb_file)
        residues = structure[0][chain_id].get_residues()
        list_residues = list(residues)
        
        dist_matrix = None
        if matrix_type=="4N4N":
            dist_matrix = self.compute_4n4n_distance_matrix(list_residues, list_residues)
        else:
            dist_matrix = self.compute_nn_distance_matrix(list_residues, list_residues, atom_1, atom_2)
        
        # if contact_map: return np.where(dist_matrix<8.0, 1, 0)
        # print(dist_matrix)
        return dist_matrix
    

# mat = DistanceMatrix()
# mat.get("data/pdbs_clean/1a5eA.pdb", "A", "NN", "CA", "CA")