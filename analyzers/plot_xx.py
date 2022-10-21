import sys
sys.path.append("../scop_classification")

import numpy as np
import pandas as pd
import utils as Utils

# read model generated outputs and converting to dataframe
outputs_dir=f"outputs/predictions/Model_contactmap_SF_512_256_8_1024_5_0.1_0.0001_1000_64_True_cuda"
data_name="train"# train, val, test
things_saved = "last_layer_learned_rep" # ["y_pred_distribution", "last_layer_learned_rep", "all_layers_attn_weights", "embeddings"]
model_outputs = Utils.load_pickle(f"{outputs_dir}/train_{things_saved}.pkl")
# print(model_outputs[0]["last_layer_learned_rep"])

df = pd.read_csv("data/splits/train_24538.txt")  #train_24538, test_5862, val_4458
reps, labels = [], []
for i in range(df.shape[0]):
    reps.append(model_outputs[i]["last_layer_learned_rep"]) 
    labels.append(df.loc[i, "SF"])

    # if i==1000: break

reps, labels = np.array(reps), np.array(labels)
print(reps.shape, labels.shape)


# tsne computation
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, metric="cosine", n_iter=2000)
tsne_results = tsne.fit_transform(reps)

tsne_one = tsne_results[:,0]
tsne_two = tsne_results[:,1]

import matplotlib.pyplot as plt
for label in labels:
    label_indices = np.where(labels==label)[0]
    if len(label_indices)>=30:
        plt.scatter(x=tsne_one[label_indices], y=tsne_two[label_indices], label=label, marker=".", alpha=0.7)
plt.show()