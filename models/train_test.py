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
max_len=1024
dim_embed=64
n_attn_heads=16 #dim_embed must be divisible by num_head
dim_ff=8*dim_embed #512
n_encoder_layers=6
dropout=0.3
init_lr=0.001
n_epochs=300 #300
batch_size=50 #50
start_epoch=1
device = "cuda" if torch.cuda.is_available() else "cpu"
out_filename = f"FullContactMapWithEmbedding_task{task}_max_len{max_len}_dim_embed{dim_embed}_n_attn_heads{n_attn_heads}_dim_ff{dim_ff}_n_encoder_layers{n_encoder_layers}_dropout{dropout}_init_lr{init_lr}_n_epochs{n_epochs}_batch_size{batch_size}"
print(out_filename)


all_data_file_path="data/splits/all_cleaned.txt"
train_data_file_path="data/splits/train_24538.txt"
val_data_file_path="data/splits/val_4458.txt"

# generating class dictionary
df = pd.read_csv(all_data_file_path)
x = df[task].unique().tolist()
class_dict = {j:i for i,j in enumerate(x)}
n_classes = len(class_dict)

# model, optimizer, scheduler, criterion, summarywriter
model = ContextTransformer.build_model(dim_embed, dim_ff, n_attn_heads, n_encoder_layers, n_classes, dropout)
model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
writer = SummaryWriter(f"outputs/tensorboard_runs/{out_filename}")

# dataset and dataloader
train_dataset = SCOPDataset(train_data_file_path, class_dict, n_attn_heads, task, max_len)
val_dataset = SCOPDataset(val_data_file_path, class_dict, n_attn_heads, task, max_len)
train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

# load the AUC/loss based model checkpoint 
if os.path.exists(f"outputs/models/{out_filename}.pth"):
    checkpoint = torch.load(f"outputs/models/{out_filename}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    prev_n_epochs = checkpoint['epoch']
    print(f"Previously trained for {prev_n_epochs} number of epochs...")
    
    # train for more epochs with new lr
    start_epoch = prev_n_epochs+1
    n_epochs = 10
    new_lr=0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=new_lr, weight_decay=5e-4)
    print(f"Train for {n_epochs} more epochs...")

best_loss = np.inf
for epoch in range(start_epoch, n_epochs+start_epoch):
    train_loss = ContextTransformer.train(model, optimizer, criterion, train_loader, device)
    val_loss, avg_acc = ContextTransformer.test(model, criterion, val_loader, device)
    crnt_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch: {epoch:03d}, crnt_lr: {crnt_lr:.5f}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, avg_acc: {avg_acc:.2f}")
    writer.add_scalar('train loss',train_loss,epoch)
    writer.add_scalar('val loss',val_loss,epoch)
    writer.add_scalar('avg_acc',avg_acc,epoch)
    writer.add_scalar('crnt_lr',crnt_lr,epoch)

    # save model dict
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    }, f"outputs/models/{out_filename}.pth")
                    