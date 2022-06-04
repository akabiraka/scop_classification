import sys
sys.path.append("../scop_classification")

from pathlib import Path
import pandas as pd
import numpy as np
import torch
import utils as Utils

import models.ContextTransformer as ContextTransformer
from models.SCOPDataset import SCOPDataset
from torch.utils.data import DataLoader

# hyperparameters
task="SF"
max_len=512 #512
dim_embed=32 #256
n_attn_heads=8 #8 #dim_embed must be divisible by num_head
dim_ff=4*dim_embed 
n_encoder_layers=5 #5
dropout=0.1
init_lr=1e-4
n_epochs=1000 #1000 
batch_size=16 #64
start_epoch=1
include_embed_layer=True
attn_type="nobackbone" #contactmap, nobackbone, longrange, distmap, noattnmask
apply_attn_mask=False if attn_type=="noattnmask" else True
apply_neighbor_aggregation=False
device = "cuda" if torch.cuda.is_available() else "cpu" # "cpu"#
out_filename = f"LocalModel_{attn_type}_{task}_{max_len}_{dim_embed}_{n_attn_heads}_{dim_ff}_{n_encoder_layers}_{dropout}_{init_lr}_{n_epochs}_{batch_size}_{include_embed_layer}_{device}_{apply_neighbor_aggregation}"
print(out_filename)
# LocalModel_nobackbone_SF_512_32_8_128_5_0.1_0.0001_1000_16_True_cuda_False


all_data_file_path="data/splits/debug/all_cleaned.txt" # debugging file paths
# all_data_file_path="data/splits/all_cleaned.txt" # real file paths

# generating class dictionary
df = pd.read_csv(all_data_file_path)
x = df[task].unique().tolist()
class_dict = {j:i for i,j in enumerate(x)}
n_classes = len(class_dict)
print(f"n_classes: {n_classes}")

# model
model = ContextTransformer.build_model(max_len, dim_embed, dim_ff, n_attn_heads, n_encoder_layers, n_classes, dropout, include_embed_layer, apply_attn_mask, apply_neighbor_aggregation)
model.to(device)

# loading learned weights
checkpoint = torch.load(f"outputs/models/{out_filename}.pth")
model.load_state_dict(checkpoint['model_state_dict'])
print(model)

x = np.zeros([1, max_len])
x[0, :20] = range(1, 21)
x = torch.tensor(x, dtype=torch.float32, device=device)

embeddings = model.get_embeddings(x)
Utils.save_as_pickle(embeddings, f"outputs/predictions/{out_filename}/20_aa_embeddings.pkl")