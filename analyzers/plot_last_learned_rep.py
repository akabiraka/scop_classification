from cProfile import label
import sys
from matplotlib import markers

from matplotlib.pyplot import bar_label
sys.path.append("../scop_classification")
import utils as Utils
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib


def plot_tsne(features, colors, img_name=None):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500)
    tsne_results = tsne.fit_transform(features)

    tsne_one = tsne_results[:,0]
    tsne_two = tsne_results[:,1]
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(x=tsne_one, y=tsne_two, c=colors, marker=".")
    ax.set_xlabel('Component-one')
    ax.set_ylabel('Component-two')
    # ax.legend()

    if img_name:
        plt.savefig(f"outputs/images/{img_name}.png", dpi=300, format="png", bbox_inches='tight', pad_inches=0.0)
        plt.close()
    else:
        plt.show()

def plot_pca(x, colors):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(x)
    print("PCA:", pca.explained_variance_ratio_)

    pca_one = pca_result[:, 0]
    pca_two = pca_result[:, 1] 


    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(x=pca_one, y=pca_two, c=colors, marker=".")
    ax.set_xlabel('PCA-1')
    ax.set_ylabel('PCA-2')

    plt.show()



dir="outputs/predictions/"
model_name="Model_contactmap_SF_512_256_8_1024_5_0.1_0.0001_1000_64_True_cuda"
data_name="val"

model_outputs = Utils.load_pickle(f"{dir}{model_name}_outputs_on_{data_name}.pkl")

# reading 1st example from the model_outputs as test
# print(model_outputs[0]["y_true"])
# print(model_outputs[0]["y_pred_distribution"])
# print(model_outputs[0]["last_layer_learned_rep"])

df = pd.DataFrame(model_outputs)
th=40
cls_indices_having_at_least_th_datams = np.stack((df["y_true"].value_counts()>=th)[:7].keys()) # taking 10 such class indices


new_df = pd.DataFrame(columns=df.columns)

for idx in cls_indices_having_at_least_th_datams:
    print(idx)
    sampled_df = df[df["y_true"]==idx].sample(n=th, random_state=1)
    new_df = new_df.append(sampled_df, ignore_index=True)
    new_df.reset_index(drop=True, inplace=True)

# print(np.array(new_df["last_layer_learned_rep"].to_list()))

x = np.array(new_df["last_layer_learned_rep"].to_list())
print(x.shape)



colors = ['red','green','blue','cyan', "magenta", "yellow", "purple", "pink", "brown", "orange"]

cls_idx_dict = {cls_idx:i for i, cls_idx in enumerate(cls_indices_having_at_least_th_datams)}

colors = []
for i, row in new_df.iterrows():
    # print(int(new_df.loc[i, "y_true"]))
    colors.append(cls_idx_dict.get(int(new_df.loc[i, "y_true"])))

# labels = np.stack(new_df["y_true"].unique())
# plot_pca(x, colors)
plot_tsne(x, colors, img_name="colocalization_of_classes")
