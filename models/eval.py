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
n_attn_heads=4 #16 #dim_embed must be divisible by num_head
dim_ff=4*dim_embed 
n_encoder_layers=3
dropout=0.1
init_lr=1e-5
n_epochs=1000 #1000 
batch_size=64 #100
start_epoch=1
include_embed_layer=True
attn_type="contactmap" #contactmap, nobackbone, longrange
device = "cuda" if torch.cuda.is_available() else "cpu" # "cpu"#
out_filename = f"CT_{attn_type}_{task}_{max_len}_{dim_embed}_{n_attn_heads}_{dim_ff}_{n_encoder_layers}_{dropout}_{init_lr}_{n_epochs}_{batch_size}_{include_embed_layer}_{device}"
print(out_filename)
# CT_contactmap_SF_512_128_4_512_3_0.1_1e-05_1000_64_True_cuda


all_data_file_path="data/splits/all_cleaned.txt"
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
criterion = torch.nn.CrossEntropyLoss()

# loading learned weights
checkpoint = torch.load(f"outputs/models/{out_filename}.pth")
model.load_state_dict(checkpoint['model_state_dict'])


# evaluating validation set
val_data_file_path="data/splits/val_4458.txt"
val_dataset = SCOPDataset(val_data_file_path, class_dict, n_attn_heads, task, max_len, attn_type)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
print(f"val data: {len(val_loader)}")
val_loss, metrics = ContextTransformer.test(model, criterion, val_loader, device)
print(f"Val: {val_loss}, {metrics}")

# evaluating test set
test_data_file_path="data/splits/test_5862.txt"
test_dataset = SCOPDataset(test_data_file_path, class_dict, n_attn_heads, task, max_len, attn_type)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print(f"val data: {len(test_loader)}")
test_loss, metrics = ContextTransformer.test(model, criterion, test_loader, device)
print(f"Test: {val_loss}, {metrics}")
