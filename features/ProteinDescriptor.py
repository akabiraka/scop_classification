import sys
sys.path.append("../scop_classification")
import features.static as STATIC
from features.SS_SA_RASA import SS_SA_RASA
from features.AAWaveEncoding import AAWaveEncoding
from objects.Onehot import Onehot
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one

class ProteinDescriptor(object):
    def __init__(self) -> None:
        super(ProteinDescriptor, self).__init__()
        self.ss_sa_rasa = SS_SA_RASA()
        self.onehot = Onehot()
        
        # computing AA wave enc memory
        self.AA_ENCODED_MEMORY = AAWaveEncoding().AA_ENCODED_MEMORY
        
        # getting static index values
        self.AA_FORMAL_CHARGE = self.__normalize(STATIC.AA_FORMAL_CHARGE)
        self.NORMALIZED_VAN_DER_WAALS_VOL = self.__normalize(STATIC.NORMALIZED_VAN_DER_WAALS_VOL)
        self.KYTE_HYDROPATHY_INDEX = self.__normalize(STATIC.KYTE_HYDROPATHY_INDEX)
        self.STERIC_PARAMETER = self.__normalize(STATIC.STERIC_PARAMETER)
        self.POLARITY = self.__normalize(STATIC.POLARITY)
        self.RASA_TRIPEPTIDE = self.__normalize(STATIC.RASA_TRIPEPTIDE)
        self.RESIDUE_VOLUME = self.__normalize(STATIC.RESIDUE_VOLUME)
        # print(self.AA_FORMAL_CHARGE)
        

    def get_node_features(self, cln_pdb_file, chain_id, residue_id):
        residue = PDBParser(QUIET=True).get_structure(" ", cln_pdb_file)[0][chain_id][residue_id]
        resname_1_letter = three_to_one(residue.resname)

        # amino acid encoding
        aa_enc = self.AA_ENCODED_MEMORY[resname_1_letter]
        onehot_enc = self.onehot.an_amino_acid(resname_1_letter, smooth=True)

        # static features
        # formal_charge = self.AA_FORMAL_CHARGE[resname_1_letter]
        normalized_van_der_waals_vol = self.NORMALIZED_VAN_DER_WAALS_VOL[resname_1_letter]
        hydropathy = self.KYTE_HYDROPATHY_INDEX[resname_1_letter]
        steric = self.STERIC_PARAMETER[resname_1_letter]
        polarity = self.POLARITY[resname_1_letter]
        rasa_tripeptide = self.RASA_TRIPEPTIDE[resname_1_letter]
        vol = self.RESIDUE_VOLUME[resname_1_letter]

        # secondary structure, solvent accessible surface area
        ss, sasa = self.ss_sa_rasa.get_ss_and_rasa_at_residue(cln_pdb_file, chain_id, residue_id)

        features = np.array([normalized_van_der_waals_vol, hydropathy, steric, polarity, rasa_tripeptide, vol, sasa], dtype=np.float32)
        features = np.hstack([features, ss, aa_enc, onehot_enc])
        # print(features)
        return features

    def get_all_node_features(self, cln_pdb_file, chain_id):
        chain = PDBParser(QUIET=True).get_structure(" ", cln_pdb_file)[0][chain_id]
        features = []
        for residue in chain.get_residues():
            x = self.get_node_features(cln_pdb_file, chain_id, residue.id)
            features.append(x)

        features = np.array(features, dtype=np.float32)
        # print(features.shape)
        return features
    
    def __normalize(self, mydict):
        # new=(x-min)/(max-min)
        return {key: (val-min(mydict.values()))/(max(mydict.values())-min(mydict.values()))for key, val in mydict.items()}


# pd = ProteinDescriptor()
# pd.get_node_features("data/pdbs_clean/1a2oA1-140.pdb", "A", (" ", 1, " "))
# pd.get_all_node_features("data/pdbs_clean/1a2oA1-140.pdb", "A")