import sys
sys.path.append("../scop_classification")
import pandas as pd
import os

class IGenerator(object):
    def __init__(self) -> None:
        pass

    def do(self, pdb_id, chain_id):
        raise NotImplementedError()

    def do_linear(self, df, n_rows_to_skip, n_rows_to_evalutate, out_file_path=None):
        new_df = pd.DataFrame()
        for i, row in df.iterrows():
            if i+1 <= n_rows_to_skip: continue
            pdb_id, chain_and_region = row["FA-PDBID"].lower(), row["FA-PDBREG"]
            if chain_and_region.find(",")!=-1: continue
            chain_and_region = chain_and_region.split(":")
            chain_id, region = chain_and_region[0], chain_and_region[1]
            if len(chain_id)>1: continue
            if pdb_id=="3iyo": continue
            
            print(f"Row:{i+1} -> {pdb_id}:{chain_id}")

            self.do(pdb_id, chain_id, region)
            
            new_df = new_df.append(df.loc[i], ignore_index=True)
            print()
            if i+1 == n_rows_to_skip+n_rows_to_evalutate: break
        if out_file_path!=None: 
            if os.path.exists(out_file_path):
                prev_df = pd.read_csv(out_file_path)
                new_df = prev_df.append(new_df, ignore_index=True)
                new_df.reset_index(drop=True, inplace=True)
            new_df.to_csv(out_file_path, index=False)
    
    def do_distributed(self, i, df):
        row = df.loc[i]
        pdb_id, chain_and_region = row["FA-PDBID"].lower(), row["FA-PDBREG"]
        if chain_and_region.find(",")!=-1: return
        chain_and_region = chain_and_region.split(":")
        chain_id, region = chain_and_region[0], chain_and_region[1]
        if len(chain_id)>1: return
        if pdb_id == "3iyo": return
        if chain_and_region.find(",")==-1: return 
        print(f"Row:{i} -> {pdb_id}:{chain_id}")
        self.do(pdb_id, chain_id, region)

# gen = IGenerator()
# gen.do(4rek, A)