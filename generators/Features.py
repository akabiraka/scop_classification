import sys
sys.path.append("../scop_classification")

import os
import pandas as pd
from objects.DistanceMatrix import DistanceMatrix
from generators.IGenerator import IGenerator
from features.ProteinDescriptor import ProteinDescriptor
import utils as Utils


class Features(IGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.pdbs_clean_dir = "data/pdbs_clean/"
        self.fastas_dir = "data/fastas/"
        self.features_dir = "data/features/"
        self.contact_maps_dir = "data/contact_maps/"
        self.distance_matrix = DistanceMatrix()
        self.protein_descriptor = ProteinDescriptor()
        

    def do(self, pdb_id, chain_id, region):
        # creating necessary file paths
        cln_pdb_file = self.pdbs_clean_dir+pdb_id+chain_id+region+".pdb"
        fasta_file = self.fastas_dir+pdb_id+chain_id+region+".fasta"
        out_feature_file = self.features_dir+pdb_id+chain_id+region+".pkl"
        out_contact_map_file = self.contact_maps_dir+pdb_id+chain_id+region+".pkl"

        # contact_map feature
        # if os.path.exists(out_contact_map_file): 
        #     contact_map = Utils.load_pickle(out_contact_map_file)
        # else:
        contact_map = self.distance_matrix.get(cln_pdb_file, chain_id, matrix_type="NN", atom_1="CA", atom_2="CA", contact_map=False)
        Utils.save_as_pickle(contact_map, out_contact_map_file)
        print(f"    Contact-map: {contact_map.shape}")

        # amino acid one-hot feature
        if os.path.exists(out_feature_file):
            features = Utils.load_pickle(out_feature_file)
        else:
            features = self.protein_descriptor.get_all_node_features(cln_pdb_file, chain_id)
            Utils.save_as_pickle(features, out_feature_file)
        print(f"    Features: {features.shape}")

        if contact_map.shape[0]!=features.shape[0]: 
            raise Exception(f"{contact_map.shape[0]}!={features.shape[0]} does not match")

inp_file_path = "data/splits/cleaned_after_pdbs_downloaded.txt"
out_file_path = "data/splits/cleaned_after_feature_computation.txt"
df = pd.read_csv(inp_file_path)
cd = Features()

n_rows_to_skip = 0
n_rows_to_evalutate = 35000
cd.do_linear(df, n_rows_to_skip, n_rows_to_evalutate, out_file_path)

# i = 1 #0-based index
# if "SLURM_ARRAY_TASK_ID" in os.environ:
#     i = int(os.environ["SLURM_ARRAY_TASK_ID"]) 
# cd.do_distributed(i, df, out_file_path)
