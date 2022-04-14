import sys
sys.path.append("../scop_classification")

import os
import pandas as pd
from objects.PDBData import PDBData
from objects.Selector import ChainAndRegionSelect
from generators.IGenerator import IGenerator

class DownloadCleanFasta(IGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.pdbs_dir = "data/pdbs/"
        self.pdbs_clean_dir = "data/pdbs_clean/"
        self.fastas_dir = "data/fastas/"
        self.pdbdata = PDBData(pdb_dir=self.pdbs_dir)

    def do(self, pdb_id, chain_id, region):
        # creating necessary file paths
        pdb_file = self.pdbs_dir+pdb_id+".cif"
        cln_pdb_file = self.pdbs_clean_dir+pdb_id+chain_id+region+".pdb"
        fasta_file = self.fastas_dir+pdb_id+chain_id+region+".fasta"
        
        # download, clean and fasta generation
        self.pdbdata.download_structure(pdb_id)
        self.pdbdata.clean(pdb_file, cln_pdb_file, selector=ChainAndRegionSelect(chain_id, region))
        self.pdbdata.generate_fasta_from_pdb(pdb_id, chain_id, input_pdb_filepath=cln_pdb_file, out_fasta_file=fasta_file)


inp_file_path = "data/splits/all.txt"
out_file_path = "data/splits/all_clean.txt"
n_rows_to_skip = 12
n_rows_to_evalutate = 5
df = pd.read_csv(inp_file_path)

i = 11 #0-based index
if "SLURM_ARRAY_TASK_ID" in os.environ:
    i = int(os.environ["SLURM_ARRAY_TASK_ID"]) 

dcf = DownloadCleanFasta()
dcf.do_linear(df, n_rows_to_skip, n_rows_to_evalutate, out_file_path)
# dcf.do_distributed(i, df)