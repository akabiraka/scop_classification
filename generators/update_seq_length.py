import sys
sys.path.append("../scop_classification")
import pandas as pd
from Bio import SeqIO

def update_seq_length(inp_file):
    df = pd.read_csv(inp_file)
    for i, row in df.iterrows():
        pdb_id, chain_and_region = row["FA-PDBID"].lower(), row["FA-PDBREG"].split(":")
        chain_id, region = chain_and_region[0], chain_and_region[1]
        fasta_file = "data/fastas/"+pdb_id+chain_id+region+".fasta"
        seq_record = next(SeqIO.parse(fasta_file, "fasta"))
        # print(seq_record)
        df.loc[i, "len"] = str(len(seq_record.seq))
        # break
    df.to_csv(inp_file, index=False)

update_seq_length("data/splits/cleaned_after_pdbs_downloaded.txt")