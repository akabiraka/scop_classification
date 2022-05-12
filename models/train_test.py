import sys
sys.path.append("../scop_classification")
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

import models.ContextTransformer as ContextTransformer
from models.SCOPDataset import SCOPDataset
torch.cuda.empty_cache()

# hyperparameters
task="SF"
max_len=512 #512
dim_embed=128 #128
n_attn_heads=8 #8 #dim_embed must be divisible by num_head
dim_ff=4*dim_embed 
n_encoder_layers=5 #5
dropout=0.1
init_lr=1e-4
n_epochs=1000 #1000 
batch_size=128 #64
start_epoch=1
include_embed_layer=True
attn_type="contactmap" #contactmap, nobackbone, longrange
device = "cuda" if torch.cuda.is_available() else "cpu" # "cpu"#
out_filename = f"Model_{attn_type}_{task}_{max_len}_{dim_embed}_{n_attn_heads}_{dim_ff}_{n_encoder_layers}_{dropout}_{init_lr}_{n_epochs}_{batch_size}_{include_embed_layer}_{device}"
print(out_filename)


# all_data_file_path="data/splits/debug/all_cleaned.txt"
# train_data_file_path="data/splits/debug/train_70.txt"
# val_data_file_path="data/splits/debug/val_14.txt"

all_data_file_path="data/splits/all_cleaned.txt"
train_data_file_path="data/splits/train_24538.txt"
val_data_file_path="data/splits/val_4458.txt"

# generating class dictionary
df = pd.read_csv(all_data_file_path)
x = df[task].unique().tolist()
class_dict = {j:i for i,j in enumerate(x)}
n_classes = len(class_dict)
print(f"n_classes: {n_classes}")

# computing class weights from the train data
train_df = pd.read_csv(train_data_file_path)
class_weights = compute_class_weight("balanced", classes=train_df[task].unique(), y=train_df[task].to_numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
# print(train_df[task].value_counts(sort=False))
print(f"class_weights: {class_weights}")

# model
model = ContextTransformer.build_model(max_len, dim_embed, dim_ff, n_attn_heads, n_encoder_layers, n_classes, dropout, include_embed_layer)
model.to(device)
trainable = ContextTransformer.count_parameters(model)
print(f"trainable weights: {trainable}")

# # optimizer, scheduler, criterion, summarywriter
optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=0.01)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
writer = SummaryWriter(f"outputs/tensorboard_runs/{out_filename}")

# # dataset and dataloader
train_dataset = SCOPDataset(train_data_file_path, class_dict, n_attn_heads, task, max_len, attn_type)
val_dataset = SCOPDataset(val_data_file_path, class_dict, n_attn_heads, task, max_len, attn_type)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

# load the AUC/loss based model checkpoint 
if os.path.exists(f"outputs/models/{out_filename}.pth"):
    checkpoint = torch.load(f"outputs/models/{out_filename}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    prev_n_epochs = checkpoint['epoch']
    print(f"Previously trained for {prev_n_epochs} number of epochs...")
    
    # train for more epochs with new lr
    start_epoch = prev_n_epochs+1
    n_epochs = 1000
    new_lr=1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=new_lr, weight_decay=0.01)
    print(f"Train for {n_epochs} more epochs...")

best_loss = np.inf
for epoch in range(start_epoch, n_epochs+start_epoch):
    train_loss = ContextTransformer.train(model, optimizer, criterion, train_loader, device)
    val_loss, metrics = ContextTransformer.test(model, criterion, val_loader, device)

    crnt_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch: {epoch:03d}, crnt_lr: {crnt_lr:.5f}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, acc: {metrics['acc']:.3f}")
    writer.add_scalar('train loss',train_loss,epoch)
    writer.add_scalar('val loss',val_loss,epoch)
    writer.add_scalar('acc',metrics["acc"],epoch)
    writer.add_scalar('precision',metrics["precision"],epoch)
    writer.add_scalar('recall',metrics["recall"],epoch)
    writer.add_scalar('crnt_lr',crnt_lr,epoch)

    # save model dict
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    }, f"outputs/models/{out_filename}.pth")
                    
