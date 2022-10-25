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

# this can be used too
# model_outputs = Utils.load_pickle("outputs/predictions/Model_contactmap_SF_512_256_8_1024_5_0.1_0.0001_1000_64_True_cuda_outputs_on_val.pkl") # for test and val
# print(model_outputs[0]["last_layer_learned_rep"])

df = pd.read_csv("data/splits/val_4458.txt")  #train_24538, test_5862, val_4458
reps, labels = [], []
for i in range(df.shape[0]):
    reps.append(model_outputs[i]["last_layer_learned_rep"]) 
    labels.append(df.loc[i, "SF"])

    # if i==1000: break

reps, labels = np.array(reps), np.array(labels)
print(reps.shape, labels.shape)

number_of_examples_to_consider_per_label = 30

unique_labels = np.unique(labels)
print(unique_labels.shape)
label_indices_to_consider = []
for label in unique_labels:
    label_indices = np.where(labels==label)[0]
    if len(label_indices)>=number_of_examples_to_consider_per_label:
        label_indices_to_consider.append(label_indices)
        

label_indices_to_consider = np.hstack(label_indices_to_consider)
# print(label_indices_to_consider.shape)
reps, labels = reps[label_indices_to_consider], labels[label_indices_to_consider]
print(reps.shape, labels.shape)

# tsne computation
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=15, metric="cosine", n_iter=5000)
tsne_results = tsne.fit_transform(reps)

tsne_one = tsne_results[:,0]
tsne_two = tsne_results[:,1]

import matplotlib.pyplot as plt
for label in unique_labels:
    label_indices = np.where(labels==label)[0]
    if len(label_indices)>=number_of_examples_to_consider_per_label:
        plt.scatter(x=tsne_one[label_indices], y=tsne_two[label_indices], s=1, label=label, marker=".")

plt.xlabel('Component-one')
plt.ylabel('Component-two')
plt.grid(True)
# plt.show()
plt.savefig("outputs/images/trainset_embedding_using_tsne.png", dpi=300, format="png", bbox_inches='tight', pad_inches=0.0)