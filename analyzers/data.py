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
def set_height_as_bar_label(rects, ax, i):
    """Attach a text label above each bar in *rects*, displaying its height."""
    
    rect = rects[i]
    height = rect.get_height()
    ax.annotate(f"{i, height}",
                xy=(i, height+10), c="red", rotation=45)
                
    for i in range(0, len(rects), 100):
        rect = rects[i]
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() , height)),
                    # xytext=(0, 3),  # 3 points vertical offset
                    # textcoords="offset points",
                    # ha='center', va='bottom')

def plot_class_distribution(df, class_label="SF", img_format="png", set_height=False, label="image"):
    # print(df[class_label])
    x = df[class_label].value_counts(sort=True)
    print(f"classes having >=10 entries: {len(x[x>=10])}") #637
    print(f"classes having <10 entries: {len(x[x<10])}")
    i = len(x[x>=10])-1 # ith_cls_having_10_data_points

    plt.figure(figsize=(10, 6))
    rects = plt.bar(range(0, len(x)), x)
    if set_height: set_height_as_bar_label(rects, plt, i)
    
    plt.xlabel(f"Class labels (alised from 0 to {len(x)-1})")
    plt.ylabel("Numer of entries (sorted)")
    # plt.xticks([])
    plt.show()
    # plt.savefig(f"outputs/images/{label}.{img_format}", dpi=300, format=img_format, bbox_inches='tight', pad_inches=0.0)


base_file_path = "data/splits/scop-cla-latest.txt"
df = pd.read_csv(base_file_path, header=None, delimiter=" ")
print(f"num of entries in base dataset: {len(df)}")

inp_file_path = "data/splits/all_cleaned.txt"
class_label="SF"
df = pd.read_csv(inp_file_path, dtype={class_label: object})
print_class_distribution(df)
plot_class_distribution(df, class_label, "png", set_height=True, label=f"{class_label}_distribution_sorted")

# the following can be used to see the train/val/test data distribution
# inp_file_path = "data/splits/test_5862.txt"
# class_label="SF"
# df = pd.read_csv(inp_file_path, dtype={class_label: object})
# print_class_distribution(df)
# plot_class_distribution(df, class_label, "png", set_height=True, label="test_distribution_sorted")