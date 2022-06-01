import sys
sys.path.append("../scop_classification")
import os
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
dim_embed=256 #256
n_attn_heads=8 #8 #dim_embed must be divisible by num_head
dim_ff=4*dim_embed 
n_encoder_layers=5 #5
dropout=0.1
init_lr=1e-4
n_epochs=1000 #1000 
batch_size=64 #64
start_epoch=1
include_embed_layer=True
attn_type="noattnmask" #contactmap, nobackbone, longrange, distmap, noattnmask
apply_attn_mask=False if attn_type=="noattnmask" else True
apply_neighbor_aggregation=False
device = "cuda" if torch.cuda.is_available() else "cpu" # "cpu"#
out_filename = f"Model_{attn_type}_{task}_{max_len}_{dim_embed}_{n_attn_heads}_{dim_ff}_{n_encoder_layers}_{dropout}_{init_lr}_{n_epochs}_{batch_size}_{include_embed_layer}_{device}_{apply_neighbor_aggregation}"
print(out_filename)
# Model_noattnmask_SF_512_256_8_1024_5_0.1_0.0001_1000_64_True_cuda_False


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
trainable_weights = ContextTransformer.count_parameters(model)
print(f"trainable weights: {trainable_weights}")
criterion = torch.nn.CrossEntropyLoss()

# loading learned weights
checkpoint = torch.load(f"outputs/models/{out_filename}.pth")
model.load_state_dict(checkpoint['model_state_dict'])


@torch.no_grad()
def test(model, criterion, loader, device):
    model.eval()
    losses, true_labels, pred_class_distributions = [], [], []
    for i, (data, y_true) in enumerate(loader):
        x, key_padding_mask, attn_mask = data["src"].to(device), data["key_padding_mask"].to(device), data["attn_mask"].to(device)
        attn_mask = torch.cat([i for i in attn_mask])
        model.zero_grad(set_to_none=True)

        true_labels.append(y_true.cpu().numpy())
        
        y_pred, last_layer_learned_rep = model(x, key_padding_mask, attn_mask)
        y_pred_distribution = torch.nn.functional.softmax(y_pred)
        pred_class_distributions.append(y_pred_distribution.squeeze(0).cpu().numpy())
        #print(y_pred_distribution.shape, pred_class_distributions)
        
        loss = criterion(y_pred, y_true.to(device))
        losses.append(loss.item())
        #break

    return {"trainable_weights": trainable_weights,
            "loss": np.mean(losses),
            "true_labels": true_labels,
            "pred_class_distributions": pred_class_distributions}


# evaluating validation set
val_data_file_path="data/splits/val_4458.txt"
val_dataset = SCOPDataset(val_data_file_path, class_dict, n_attn_heads, task, max_len, attn_type)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
print(f"val data: {len(val_loader)}")
metrics = test(model, criterion, val_loader, device)
print(f"Val: {metrics}")
Utils.save_as_pickle(metrics, f"outputs/predictions/{out_filename}_val_result.pkl")


# evaluating test set
test_data_file_path="data/splits/test_5862.txt"
test_dataset = SCOPDataset(test_data_file_path, class_dict, n_attn_heads, task, max_len, attn_type)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print(f"val data: {len(test_loader)}")
metrics = test(model, criterion, test_loader, device)
print(f"Test: {metrics}")
Utils.save_as_pickle(metrics, f"outputs/predictions/{out_filename}_test_result.pkl")
