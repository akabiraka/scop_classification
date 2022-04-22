import sys
sys.path.append("../scop_classification")
import pandas as pd
from Bio import SeqIO

def separate_classes(inp_file, out_file):
    df = pd.read_csv(inp_file, delim_whitespace=True)
    for i, row in df.iterrows():
        # print(row)
        x = row["SCOPCLA"].split(",")
        TP, CL, CF, SF, FA = x[0][3:], x[1][3:], x[2][3:], x[3][3:], x[4][3:]
        df.loc[i, "TP"] = TP 
        df.loc[i, "CL"] = CL 
        df.loc[i, "CF"] = CF 
        df.loc[i, "SF"] = SF
        df.loc[i, "FA"] = FA 
        # break
    df.to_csv(out_file, index=False)

separate_classes("data/splits/scop-cla-latest.txt", "data/splits/cleaned_after_separating_class_labels.txt")