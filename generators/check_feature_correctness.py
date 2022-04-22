import sys
sys.path.append("../scop_classification")
import pandas as pd
import os
from generators.IGenerator import IGenerator
import utils as Utils


class CheckFeatureCorrectness(IGenerator):
    def __init__(self) -> None:
        super(CheckFeatureCorrectness, self).__init__()
        self.features_dir = "data/features/"
        self.distance_matrices_dir = "data/distance_matrices/"
    
    def do(self, pdb_id, chain_id, region):
        feature_file = self.features_dir+pdb_id+chain_id+region+".pkl"
        dist_mat_file = self.distance_matrices_dir+pdb_id+chain_id+region+".pkl"

        if not os.path.exists(dist_mat_file): 
            raise Exception(f"{dist_mat_file} does not exists.")
        
        if not os.path.exists(feature_file): 
            raise Exception(f"{feature_file} does not exists.")
        
        dist_mat = Utils.load_pickle(dist_mat_file)
        features = Utils.load_pickle(feature_file)
        if dist_mat.shape[0]!=features.shape[0]: 
            raise Exception(f"dist_mat_file and feature_file shape does not match: {dist_mat.shape[0]}!={features.shape[0]}")
        
inp_file_path = "data/splits/all_cleaned.txt"
df = pd.read_csv(inp_file_path)
cfc = CheckFeatureCorrectness()

n_rows_to_skip = 0
n_rows_to_evalutate = 35000
cfc.do_linear(df, n_rows_to_skip, n_rows_to_evalutate, None)