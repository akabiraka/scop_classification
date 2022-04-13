import sys
sys.path.append("../scop_classification")
import pandas as pd

class IGenerator(object):
    def __init__(self) -> None:
        pass

    def do(self, pdb_id, chain_id):
        raise NotImplementedError()

    def do_linear(self, df, n_rows_to_skip, n_rows_to_evalutate, out_file_path=None):
        new_df = pd.DataFrame()
        for i, row in df.iterrows():
            if i+1 <= n_rows_to_skip: continue
            pdb_id, chain_and_region = row["FA-PDBID"].lower(), row["FA-PDBREG"].split(":")
            chain_id, region = chain_and_region[0], chain_and_region[1]
            if len(chain_id)>1: continue
            if pdb_id == "3iyo": continue
            print(f"Row:{i+1} -> {pdb_id}:{chain_id}")

            self.do(pdb_id, chain_id, region)
            
            new_df = new_df.append(df.loc[i], ignore_index=True)
            print()
            if i+1 == n_rows_to_skip+n_rows_to_evalutate: break
        if out_file_path is not None: 
            new_df.to_csv(out_file_path, index=False)
    
    def do_distributed(self, i, df):
        row = df.loc[i]
        pdb_id, chain_id = row.PDBchain[:4].lower(), row.PDBchain[4:]
        if len(chain_id)>1: return
        print(f"Row:{i} -> {pdb_id}:{chain_id}")
        self.do(pdb_id, chain_id)

# gen = IGenerator()
# gen.do(4rek, A)