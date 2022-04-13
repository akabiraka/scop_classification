import sys
sys.path.append("../scop_classification")

import numpy as np

class Onehot(object):
    def __init__(self) -> None:
        self.AA = "ARNDCQEGHILKMFPSTWYV"
        self.SS = "HBC" # H:helix, B:beta, C:coil
        self.SS_dict = {"H":"H", "G":"H", "I":"H", "B":"B", "E":"B", "T":"C", "S":"C", "-":"C"} # helix:H,G,I; sheet:B,E; coil:T,S,-
        self.SA = "BEI" # B:buried, E:exposed, I:intermediate

    def an_amino_acid(self, aa, smooth=True):
        if smooth: return np.array([0.1 if char!=aa else 0.9 for char in self.AA], dtype=np.float32)
        return np.array([0.0 if char != aa else 1.0 for char in self.AA], dtype=np.float32)

    def sequence(self, seq, smooth=True):
        features = [self.an_amino_acid(aa, smooth) for aa in seq]
        return np.array(features)

    def mutation_site_with_neighbors(self, seq, i, n_neighbors=3, smooth=True):
        """
        Args:
            seq (string): Amino acid sequence
            i (int): 0-based mutation site
            n_neighbors (int, optional): On both sides of i. Defaults to 3.
            smooth (bool, optional): The encoding type. Defaults to True.

        Returns:
            ndarray: [2*n_neighbors+1, 20]
        """
        if i-n_neighbors < 0: 
            sub_seq = seq[:i+n_neighbors+1]
            sub_seq = "0" * (2*n_neighbors+1 - len(sub_seq)) + sub_seq #padding at the front
        elif i+n_neighbors >= len(seq): 
            sub_seq = seq[i-n_neighbors:]
            sub_seq = sub_seq + "0" * (2*n_neighbors+1 - len(sub_seq)) #padding at the end
        else: sub_seq = seq[i-n_neighbors:i+n_neighbors+1]
        return self.sequence(sub_seq, smooth)

    def a_secondary_structure(self, ss, smooth=True):
        letter = self.SS_dict.get(ss)
        if smooth: return np.array([0.1 if char != letter else 0.9 for char in self.SS], dtype=np.float32)
        return np.array([0.0 if char != letter else 1.0 for char in self.SS], dtype=np.float32)

    def a_solvent_accessibility(self, sa, smooth=True):
        if smooth: return np.array([0.1 if char != sa else 0.9 for char in self.SA], dtype=np.float32)
        return np.array([0.0 if char != sa else 1.0 for char in self.SA], dtype=np.float32)


# onehot = Onehot()
# x = onehot.an_amino_acid("A", smooth=False)
# x = onehot.a_secondary_structure("G")
# x = onehot.a_solvent_accessibility("I")
# x = onehot.sequence("RVEEV", smooth=True)
# print(x)
