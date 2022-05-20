import sys
sys.path.append("../scop_classification")

import pandas as pd
import torch
import utils as Utils

import models.ContextTransformer as ContextTransformer
from models.SCOPDataset import SCOPDataset
from torch.utils.data import DataLoader

# hyperparameters
task="SF"
max_len=512 #1024
dim_embed=256 #512
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
out_filename = f"Model_{attn_type}_{task}_{max_len}_{dim_embed}_{n_attn_heads}_{dim_ff}_{n_encoder_layers}_{dropout}_{init_lr}_{n_epochs}_{batch_size}_{include_embed_layer}_{device}"
print(out_filename)
# Model_contactmap_SF_512_256_8_1024_5_0.1_0.0001_1000_64_True_cuda


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

# loading learned weights
checkpoint = torch.load(f"outputs/models/{out_filename}.pth")
model.load_state_dict(checkpoint['model_state_dict'])


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    outputs = []
    for i, (data, y_true) in enumerate(loader):
        x, key_padding_mask, attn_mask = data["src"].to(device), data["key_padding_mask"].to(device), data["attn_mask"].to(device)
        attn_mask = torch.cat([i for i in attn_mask])
        
        model.zero_grad(set_to_none=True)
        y_pred, last_layer_learned_rep = model(x, key_padding_mask, attn_mask)

        # saving per item predictions
        print(torch.nn.functional.softmax(y_pred))
        print(torch.nn.functional.softmax(y_pred).shape)
        print(torch.nn.functional.softmax(y_pred, dim=1).shape)
        print(torch.nn.functional.softmax(y_pred, dim=1))

        outputs.append({
            "y_true": y_true.unsqueeze().cpu().numpy(),
            "y_pred_distribution": torch.nn.functional.softmax(y_pred),
            "last_layer_learned_rep": last_layer_learned_rep.unsqueeze().cpu().numpy()
        })

        break

    return outputs


# evaluating validation set
val_data_file_path="data/splits/val_4458.txt"
val_dataset = SCOPDataset(val_data_file_path, class_dict, n_attn_heads, task, max_len, attn_type)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
print(f"val data: {len(val_loader)}")
outputs = test(model, val_loader, device)
print(outputs)
Utils.save_as_pickle(outputs, f"outputs/predictions/{out_filename}_outputs_on_val.pkl")


# evaluating test set
# test_data_file_path="data/splits/test_5862.txt"
# test_dataset = SCOPDataset(test_data_file_path, class_dict, n_attn_heads, task, max_len, attn_type)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
# print(f"val data: {len(test_loader)}")
# outputs = test(model, test_loader, device)
# Utils.save_as_pickle(outputs, f"outputs/predictions/{out_filename}_outputs_on_val.pkl")
