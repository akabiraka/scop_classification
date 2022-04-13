import sys
sys.path.append("../scop_classification")
import pandas as pd
import matplotlib.pyplot as plt

def print_class_distribution(inp_file):
    df = pd.read_csv(inp_file)
    n_unique_pdbs = len(df["FA-PDBID"].unique().tolist())
    n_TP = len(df["TP"].unique().tolist())
    n_CL = len(df["CL"].unique().tolist())
    n_CF = len(df["CF"].unique().tolist())
    n_SF = len(df["SF"].unique().tolist())
    n_FA = len(df["FA"].unique().tolist())
    max_len = df["len"].max()
    
    print(df.head())
    print(f"Data shape: {df.shape}") # (36534, 11)
    print(f"n_unique_pdbs: {n_unique_pdbs}, n_TP: {n_TP}, n_CL: {n_CL}, n_CF: {n_CF}, n_SF: {n_SF}, n_FA: {n_FA}")
    print(f"max-len: {max_len}")

    x = df["FA"].value_counts()
    plt.bar(range(1, len(x)+1), x)
    plt.show()

print_class_distribution("data/splits/all_10_clean.txt")