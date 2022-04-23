import sys
sys.path.append("../scop_classification")
import pandas as pd
import matplotlib.pyplot as plt

def print_class_distribution(df):
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


# 
def set_height_as_bar_label(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def plot_class_distribution(df, class_label="SF"):
    x = df[class_label].value_counts()
    rects = plt.bar(range(1, len(x)+1), x)
    # set_height_as_bar_label(rects, plt)
    # plt.show()
    plt.savefig(f"outputs/images/{class_label}_distribution.png", dpi=300, format="png", bbox_inches='tight', pad_inches=0.0)


inp_file_path = "data/splits/all_cleaned.txt"
class_label="SF"
df = pd.read_csv(inp_file_path, dtype={class_label: object})
print_class_distribution(df)
plot_class_distribution(df, class_label)