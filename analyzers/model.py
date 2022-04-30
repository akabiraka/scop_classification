import sys
sys.path.append("../scop_classification")
import torch
import models.ContextTransformer as ContextTransformer
import pandas as pd

task="SF"
max_len=1024 #1024
dim_embed=128 #512
n_attn_heads=16 #16 #dim_embed must be divisible by num_head
dim_ff=4*dim_embed 
n_encoder_layers=5
dropout=0.3
init_lr=0.001
n_epochs=1000 #1000 
batch_size=20 #100
start_epoch=1
include_embed_layer=True
attn_type="contactmap" #contactmap, nobackbone, longrange
device = "cuda" if torch.cuda.is_available() else "cpu"
out_filename = f"Model_{attn_type}_{task}_{max_len}_{dim_embed}_{n_attn_heads}_{dim_ff}_{n_encoder_layers}_{dropout}_{init_lr}_{n_epochs}_{batch_size}_{include_embed_layer}_{device}"
print(out_filename)


all_data_file_path="data/splits/all_cleaned.txt"
# generating class dictionary
df = pd.read_csv(all_data_file_path)
x = df[task].unique().tolist()
class_dict = {j:i for i,j in enumerate(x)}
n_classes = len(class_dict)
print(f"n_classes: {n_classes}")

# model
model = ContextTransformer.build_model(max_len, dim_embed, dim_ff, n_attn_heads, n_encoder_layers, n_classes, dropout, include_embed_layer)
# model.to(device)
# checkpoint = torch.load(f"outputs/models/{out_filename}.pth")
# model.load_state_dict(checkpoint['model_state_dict'])
trainable = ContextTransformer.count_parameters(model)
print(f"trainable weights: {trainable}")