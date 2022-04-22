import sys
sys.path.append("../scop_classification")
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import models.ContextTransformer as ContextTransformer
from models.SCOPDataset import SCOPDataset


# hyperparameters
task="SF"
max_len = 1024
dim_embed = 50
n_attn_heads = 10 #dim_embed must be divisible by num_head
dim_ff = 2*dim_embed
n_encoder_layers=6
dropout=0.3
init_lr = 0.001
start_epoch = 1
n_epochs = 300
batch_size = 50
device = "cuda" if torch.cuda.is_available() else "cpu"
out_filename = f"ContextTransformer_task{task}_max_len{max_len}_dim_embed{dim_embed}_n_attn_heads{n_attn_heads}_dim_ff{dim_ff}_n_encoder_layers{n_encoder_layers}_dropout{dropout}_init_lr{init_lr}_n_epochs{n_epochs}_batch_size{batch_size}"
print(out_filename)

# generating class dictionary
df = pd.read_csv("data/splits/all_cleaned.txt")
x = df[task].unique().tolist()
class_dict = {j:i for i,j in enumerate(x)}
n_classes = len(class_dict)

# model, optimizer, scheduler, criterion, summarywriter
model = ContextTransformer.build_model(dim_embed, dim_ff, n_attn_heads, n_encoder_layers, n_classes, dropout)
# ContextTransformer.init_model(model)
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
writer = SummaryWriter(f"outputs/tensorboard_runs/{out_filename}")

# datasets
train_dataset = SCOPDataset("data/splits/train_24538.txt", class_dict, n_attn_heads, task, max_len)
val_dataset = SCOPDataset("data/splits/val_4458.txt", class_dict, n_attn_heads, task, max_len)
test_dataset = SCOPDataset("data/splits/test_5862.txt", class_dict, n_attn_heads, task, max_len)
# dataloader
train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
print(len(train_loader), len(val_loader), len(test_loader))

# load the AUC/loss based model checkpoint 
if os.path.exists(f"outputs/models/{out_filename}.pth"):
    checkpoint = torch.load(f"outputs/models/{out_filename}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    pre_n_epochs = checkpoint['epoch']
    print(f"Previously trained for {pre_n_epochs} number of epochs...")
    
    # train for more epochs
    start_epoch = pre_n_epochs+1
    n_epochs = 100
    print(f"Train for {n_epochs} more epochs...")

best_loss = np.inf
for epoch in range(start_epoch, n_epochs+start_epoch):
    train_loss = ContextTransformer.train(model, optimizer, criterion, train_loader, device)
    val_loss = ContextTransformer.test(model, criterion, val_loader, device)
    crnt_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch: {epoch:03d}, crnt_lr: {crnt_lr:.5f}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")
    writer.add_scalar('train loss',train_loss,epoch)
    writer.add_scalar('val loss',val_loss,epoch)
    writer.add_scalar('crnt_lr',crnt_lr,epoch)

    # save model dict
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, f"outputs/models/{out_filename}.pth")