import sys
sys.path.append("../scop_classification")

import numpy as np
import pandas as pd
import utils as Utils

# getting superfam class ids as list
all_data_file_path="data/splits/all_cleaned.txt"
df = pd.read_csv(all_data_file_path)
superfam_clssses = df["SF"].unique().tolist()

# maps SCOP superfam class ids to class names 
df_cls_des = pd.read_csv("data/splits/scop-des-latest.txt")
scop_node_label_dict = {}
for i in range(len(df_cls_des)):
    key = int(df_cls_des.loc[i].values[0].split(" ")[0])
    label = " ".join(df_cls_des.loc[i].values[0].split(" ")[1:])
    scop_node_label_dict[key] = label
    # if i==5: break
print(len(scop_node_label_dict))


# read model generated outputs and converting to dataframe
dir="outputs/predictions/"
model_name="Model_contactmap_SF_512_256_8_1024_5_0.1_0.0001_1000_64_True_cuda"
data_name="val"
model_outputs = Utils.load_pickle(f"{dir}{model_name}_outputs_on_{data_name}.pkl")
# reading 1st example from the model_outputs as test
# print(model_outputs[0]["y_true"])
# print(model_outputs[0]["y_pred_distribution"])
# print(model_outputs[0]["last_layer_learned_rep"])
df = pd.DataFrame(model_outputs)
# print(df.head())


# creating new column with class names using the previously created dictionaries
# for i in range(len(df)):
#     df["cls_label"] = scop_node_label_dict[superfam_clssses[df.loc[0, "y_true"]]]
#     # if i==5: break

# print(df.head())


# creating a new_df having th data points of n classes
n=10
th=40
cls_indices_having_at_least_th_datams = np.stack((df["y_true"].value_counts()>=th)[:n].keys()) # taking 10 such class indices
new_df = pd.DataFrame(columns=df.columns)
for idx in cls_indices_having_at_least_th_datams:
    print(idx)
    sampled_df = df[df["y_true"]==idx].sample(n=th, random_state=1)
    new_df = new_df.append(sampled_df, ignore_index=True)
    new_df.reset_index(drop=True, inplace=True)
# print(new_df.head())

# making nd array of features from new_df
features = np.array(new_df["last_layer_learned_rep"].to_list())
print(features.shape)


# initializing the color dictionaries for each class index
colors = ['red','green','blue','cyan', "magenta", "yellow", "purple", "pink", "brown", "orange"]
cls_idx_dict = {cls_idx:i for i, cls_idx in enumerate(cls_indices_having_at_least_th_datams)}

# tsne computation
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500)
tsne_results = tsne.fit_transform(features)

tsne_one = tsne_results[:,0]
tsne_two = tsne_results[:,1]

# plotting
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)

# print(cls_indices_having_at_least_th_datams)
for _, cls_idx in enumerate(cls_indices_having_at_least_th_datams):
    # print(new_df[new_df["y_true"]==cls_idx])
    indices = new_df[new_df["y_true"]==cls_idx].index
    color = colors[cls_idx_dict.get(int(cls_idx))]
    label = scop_node_label_dict[superfam_clssses[cls_idx]]
    print(cls_idx, superfam_clssses[cls_idx], label, color)
    # print(indices)
    x_coords, y_coords = [], []
    for idx in indices:
        x_coords.append(tsne_one[idx])
        y_coords.append(tsne_two[idx])
    ax.scatter(x=x_coords, y=y_coords, c=color, label=label, marker=".", alpha=0.7)
    
    # break

ax.set_xlabel('Component-one')
ax.set_ylabel('Component-two')
ax.grid(True)
# ax.legend()
ax.legend(bbox_to_anchor=(0, -.1), loc='upper left', ncol=1)

# plt.show()
plt.savefig(f"outputs/images/colocalization_of_classes.png", dpi=300, format="png", bbox_inches='tight', pad_inches=0.0)