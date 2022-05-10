import sys
sys.path.append("../scop_classification")
import os
import pandas as pd
import numpy as np
import torch

import models.ContextTransformer as ContextTransformer
from models.SCOPDataset import SCOPDataset
from torch.utils.data import DataLoader

# hyperparameters
task="SF"
max_len=512 #1024
dim_embed=128 #512
n_attn_heads=8 #16 #dim_embed must be divisible by num_head
dim_ff=4*dim_embed 
n_encoder_layers=6
dropout=0.3
init_lr=0.001
n_epochs=1000 #1000 
batch_size=50 #100
start_epoch=1
include_embed_layer=True
attn_type="contactmap" #contactmap, nobackbone, longrange
device = "cuda" if torch.cuda.is_available() else "cpu" # "cpu"#
out_filename = f"Model_{attn_type}_{task}_{max_len}_{dim_embed}_{n_attn_heads}_{dim_ff}_{n_encoder_layers}_{dropout}_{init_lr}_{n_epochs}_{batch_size}_{include_embed_layer}_{device}"
print(out_filename)
# Model_contactmap_SF_512_128_8_512_6_0.3_0.001_1000_50_True_cuda


all_data_file_path="data/splits/all_cleaned.txt"
val_data_file_path="data/splits/val_4458.txt"

# generating class dictionary
df = pd.read_csv(all_data_file_path)
x = df[task].unique().tolist()
class_dict = {j:i for i,j in enumerate(x)}
n_classes = len(class_dict)
print(f"n_classes: {n_classes}")

# model
model = ContextTransformer.build_model(max_len, dim_embed, dim_ff, n_attn_heads, n_encoder_layers, n_classes, dropout, include_embed_layer)
model.to(device)
trainable = ContextTransformer.count_parameters(model)
print(f"trainable weights: {trainable}")

val_dataset = SCOPDataset(val_data_file_path, class_dict, n_attn_heads, task, max_len, attn_type)
val_loader = DataLoader(val_dataset, len(val_dataset), shuffle=False)

checkpoint = torch.load(f"outputs/models/{out_filename}.pth")
model.load_state_dict(checkpoint['model_state_dict'])

def test(model, loader, device):
    model.eval()
    losses, acc_list = [], []
    for i, (data, y_true) in enumerate(loader):
        x, key_padding_mask, attn_mask = data["src"].to(device), data["key_padding_mask"].to(device), data["attn_mask"].to(device)
        attn_mask = torch.cat([i for i in attn_mask])
        model.zero_grad(set_to_none=True)
        y_pred = model(x, key_padding_mask, attn_mask)
        
        print("y_pred: {y_pred}")
        print("y_true: {y_true}")
        print()
        # break


test(model, val_loader, device)