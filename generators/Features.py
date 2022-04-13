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
        contact_map = self.distance_matrix.get(cln_pdb_file, chain_id, matrix_type="NN", atom_1="CA", atom_2="CA", contact_map=True)
        Utils.save_as_pickle(contact_map, out_contact_map_file)
        print(f"    Contact-map: {contact_map.shape}")

        # amino acid one-hot feature
        features = self.protein_descriptor.get_all_node_features(cln_pdb_file, chain_id)
        Utils.save_as_pickle(features, out_feature_file)
        print(f"    Features: {features.shape}")

inp_file_path = "data/splits/all_clean.txt"
n_rows_to_skip = 0
n_rows_to_evalutate = 1000
df = pd.read_csv(inp_file_path)

i = 10 #0-based index
if "SLURM_ARRAY_TASK_ID" in os.environ:
    i = int(os.environ["SLURM_ARRAY_TASK_ID"]) 

cd = Features()
cd.do_linear(df, n_rows_to_skip, n_rows_to_evalutate)
# cd.do_distributed(i, df)
