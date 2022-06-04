import sys
sys.path.append("../scop_classification")

import numpy as np
import pandas as pd
import utils as Utils
import matplotlib.pyplot as plt
from features import static as STATIC
from models.SCOPDataset import SCOPDataset
from torch.utils.data import DataLoader

def compute_tsne(features):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500)
    tsne_results = tsne.fit_transform(features)

    tsne_one = tsne_results[:,0]
    tsne_two = tsne_results[:,1]
    return tsne_one, tsne_two



# necessary directory configs
# debug set
# all_data_file_path="data/splits/debug/all_cleaned.txt"
# data_file_path="data/splits/debug/val_14.txt" 
# model_outputs_file_path="outputs/predictions/LocalModel_nobackbone_SF_512_32_8_128_5_0.1_0.0001_1000_16_True_cuda_False/val_embeddings.pkl"
# real set
all_data_file_path="data/splits/all_cleaned.txt"
data_file_path="data/splits/val_4458.txt"
model_outputs_file_path="outputs/predictions/Model_nobackbone_SF_512_256_8_1024_5_0.1_0.0001_1000_64_True_cuda_False/val_embeddings.pkl"



# data loader configs
# hyperparameters
task="SF"
max_len=512 #512
n_encoder_layers=5
n_attn_heads=8 #8 #dim_embed must be divisible by num_head
attn_type="nobackbone" #contactmap, nobackbone, longrange, distmap, noattnmask

# generating class dictionary
df = pd.read_csv(all_data_file_path)
x = df[task].unique().tolist()
class_dict = {j:i for i,j in enumerate(x)}
n_classes = len(class_dict)
print(f"n_classes: {n_classes}")




dataset = SCOPDataset(data_file_path, class_dict, n_attn_heads, task, max_len, attn_type)
loader = DataLoader(dataset, batch_size=1, shuffle=False)
model_outputs = Utils.load_pickle(model_outputs_file_path)


color_names = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan", 
               "black", "magenta", "chocolate", "greenyellow", "teal", "deepskyblue", "blueviolet", "deeppink", "maroon", "tomato"]


for i, (data, y_true) in enumerate(loader):
    if i!=123: continue
    print(data["src"].shape, data["key_padding_mask"].shape, data["attn_mask"].shape) #to access data

    key_padding_mask = data["key_padding_mask"].squeeze(dim=0).cpu().numpy()
    seq_len = np.argmax(key_padding_mask==True)
    print(f"seq-len: {seq_len}")

    embeddings = model_outputs[i]["embeddings"][:seq_len]
    print(embeddings.shape)
    tsne_one, tsne_two = compute_tsne(embeddings)
    src = data["src"].squeeze(0).cpu().numpy()[:seq_len]
    
    # plotting
    for aa_idx in range(1, 21):
        indices = [j for j, x in enumerate(src) if x==aa_idx]
        colors = np.repeat(color_names[aa_idx-1], len(indices))
        label=STATIC.AA_NAMES[STATIC.AA_1_LETTER[aa_idx-1]]

        
        plt.scatter(x=tsne_one[indices], y=tsne_two[indices], label=label, c=colors, marker=".", alpha=0.7) #, c=aa_idx, label=aa_idx
    plt.legend(bbox_to_anchor=(.5, 1.44), loc='upper center', ncol=3)
    # plt.show()
    plt.savefig(f"outputs/images/aa_embedding_of_a_seq.png", dpi=300, format="png", bbox_inches='tight', pad_inches=0.05)

    break # since we need to plot for only one sequence
