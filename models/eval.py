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
n_encoder_layers=5
dropout=0.1
init_lr=1e-4
n_epochs=1000 #1000 
batch_size=64 #100
start_epoch=1
include_embed_layer=True
attn_type="contactmap" #contactmap, nobackbone, longrange
device = "cuda" if torch.cuda.is_available() else "cpu" # "cpu"#
out_filename = f"Model1_{attn_type}_{task}_{max_len}_{dim_embed}_{n_attn_heads}_{dim_ff}_{n_encoder_layers}_{dropout}_{init_lr}_{n_epochs}_{batch_size}_{include_embed_layer}_{device}"
print(out_filename)
# Model1_contactmap_SF_512_128_8_512_5_0.1_0.0001_1000_64_True_cuda


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


@torch.no_grad()
def test(model, criterion, loader, device):
    model.eval()
    losses, pred_labels, true_labels = [], [], []
    for i, (data, y_true) in enumerate(loader):
        x, key_padding_mask, attn_mask = data["src"].to(device), data["key_padding_mask"].to(device), data["attn_mask"].to(device)
        attn_mask = torch.cat([i for i in attn_mask])
        model.zero_grad(set_to_none=True)
        y_pred = model(x, key_padding_mask, attn_mask)
        loss = criterion(y_pred, y_true.to(device))
        
        losses.append(loss.item())
        pred_labels.append(y_pred.argmax(dim=1).cpu().numpy())
        true_labels.append(y_true.cpu().numpy())
        
    metrics = get_metrics(true_labels, pred_labels)
    loss = np.mean(losses)
    return loss, metrics


def get_metrics(target_classes, pred_classes):
    from sklearn.metrics import accuracy_score, recall_score, precision_score
    acc = accuracy_score(target_classes, pred_classes)
    precision = precision_score(target_classes, pred_classes, average="micro")
    recall = recall_score(target_classes, pred_classes, average="micro")
    return {"acc": acc, 
            "precision": precision, 
            "recall": recall, 
            "pred_classes": pred_classes, 
            "target_classes": target_classes}

# evaluating validation set
val_data_file_path="data/splits/val_4458.txt"
val_dataset = SCOPDataset(val_data_file_path, class_dict, n_attn_heads, task, max_len, attn_type)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
print(f"val data: {len(val_loader)}")
val_loss, metrics = test(model, criterion, val_loader, device)
print(f"Val: {val_loss}, {metrics}")

# evaluating test set
test_data_file_path="data/splits/test_5862.txt"
test_dataset = SCOPDataset(test_data_file_path, class_dict, n_attn_heads, task, max_len, attn_type)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print(f"val data: {len(test_loader)}")
test_loss, metrics = test(model, criterion, test_loader, device)
print(f"Test: {test_loss}, {metrics}")
