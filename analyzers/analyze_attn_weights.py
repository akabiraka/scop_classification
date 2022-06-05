import sys
sys.path.append("../scop_classification")
import itertools
import utils as Utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.SCOPDataset import SCOPDataset
from torch.utils.data import DataLoader


# debug set
all_data_file_path="data/splits/debug/all_cleaned.txt"
data_file_path="data/splits/debug/val_14.txt" 

# real set
# all_data_file_path="data/splits/all_cleaned.txt"
# val_data_file_path="data/splits/val_4458.txt"

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

dir = "outputs/predictions/"
filename = "LocalModel_nobackbone_SF_512_32_8_128_5_0.1_0.0001_1000_16_True_cuda_False_outputs_on_val"
model_outputs = Utils.load_pickle(dir+filename+".pkl")

for i, (data, y_true) in enumerate(loader):
    print(data["src"].shape, data["key_padding_mask"].shape, data["attn_mask"].shape) #to access data
    attn_weights_of_an_input = model_outputs[i]["all_layers_attn_weights"] #[n_encoder_layers, n_attn_heads, max_len, max_len]
    
    key_padding_mask = data["key_padding_mask"].squeeze(dim=0).numpy()
    seq_len = np.argmax(key_padding_mask==True)
    attn_weights_of_an_input[0, 0, :seq_len, :seq_len]

    # n_encoder_layers =1
    dist = np.zeros((n_encoder_layers*n_attn_heads, n_encoder_layers*n_attn_heads))
    for j, (l1, h1) in enumerate(itertools.product(range(n_encoder_layers), range(n_attn_heads))):
        for k, (l2, h2) in enumerate(itertools.product(range(n_encoder_layers), range(n_attn_heads))):
            m1 = attn_weights_of_an_input[l1, h1, :seq_len, :seq_len]
            m2 = attn_weights_of_an_input[l2, h2, :seq_len, :seq_len]
            dist[j, k] = np.sqrt(np.sum((m1 - m2) ** 2))
    plt.imshow(dist, interpolation='none')
    # plt.imshow(attn_weights_of_an_input[0, 0, :seq_len, :seq_len], interpolation='none')
    # plt.imshow(data["attn_mask"][0, 0].numpy(), interpolation='none')
    # plt.imshow(np.repeat(data["key_padding_mask"].numpy(), 20, axis=0), interpolation='none')
    plt.show()
    
    if i==0:
        break