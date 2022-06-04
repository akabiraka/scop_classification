import sys
sys.path.append("../scop_classification")

from pathlib import Path
import pandas as pd
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



# debugging file paths
all_data_file_path="data/splits/debug/all_cleaned.txt" 
val_data_file_path="data/splits/debug/val_14.txt" 
test_data_file_path="data/splits/debug/test_16.txt"

# real file paths
# all_data_file_path="data/splits/all_cleaned.txt"
# val_data_file_path="data/splits/val_4458.txt"
# test_data_file_path="data/splits/test_5862.txt"



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




@torch.no_grad()
def test(model, loader, device, thing_to_save, where_to_save=None):
    model.eval()
    outputs = []
    for i, (data, y_true) in enumerate(loader):
        x, key_padding_mask, attn_mask = data["src"].to(device), data["key_padding_mask"].to(device), data["attn_mask"].to(device)
        attn_mask = torch.cat([i for i in attn_mask])
        model.zero_grad(set_to_none=True)

        if thing_to_save=="all_layers_attn_weights":
            all_layers_attn_weights = model.get_all_layers_attn_weights(x, key_padding_mask, attn_mask)
            all_layers_attn_weights = all_layers_attn_weights.squeeze(dim=1).cpu().numpy() #[n_encoder_layers, n_attn_heads, max_len, max_len]
            if where_to_save is not None: 
                Path(where_to_save).mkdir(parents=True, exist_ok=True)
                Utils.save_as_pickle(all_layers_attn_weights, f"{where_to_save}/{str(i)}.pkl")
            outputs=None
        
        elif thing_to_save=="y_pred_distribution":
            y_pred = model(x, key_padding_mask, attn_mask)
            outputs.append({
                "y_true": y_true.squeeze(dim=0).cpu().numpy(), #scaler
                "y_pred_distribution": torch.nn.functional.softmax(y_pred, dim=1).squeeze(dim=0).cpu().numpy() #[n_classes]
            })
            
        elif thing_to_save=="last_layer_learned_rep":
            last_layer_learned_rep = model.get_last_layer_learned_rep(x, key_padding_mask, attn_mask)
            outputs.append({
                "last_layer_learned_rep": last_layer_learned_rep.squeeze(dim=0).cpu().numpy() #[]
            })


        elif thing_to_save=="embeddings":
            embeddings = model.get_embeddings(x)
            outputs.append({
                "embeddings": embeddings.squeeze(dim=0).cpu().numpy() #[max_len, dim_embed]
            })
            

        else:
            raise NotImplemented("The selected item is not implemented to save yet.")

        # break

    return outputs



things_to_save = ["y_pred_distribution", "last_layer_learned_rep", "all_layers_attn_weights", "embeddings"]
outputs_dir=f"outputs/predictions/{out_filename}"
Path(outputs_dir).mkdir(parents=True, exist_ok=True)



for thing_to_save in things_to_save:
    # evaluating validation set
    val_dataset = SCOPDataset(val_data_file_path, class_dict, n_attn_heads, task, max_len, attn_type)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(f"Computing '{thing_to_save}' for val data of size {len(val_loader)}")
    outputs = test(model, val_loader, device, thing_to_save, where_to_save=f"{outputs_dir}/{thing_to_save}/val")
    if outputs is not None: Utils.save_as_pickle(outputs, f"{outputs_dir}/val_{thing_to_save}.pkl")



    # evaluating test set
    test_dataset = SCOPDataset(test_data_file_path, class_dict, n_attn_heads, task, max_len, attn_type)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Computing '{thing_to_save}' for val data of size {len(test_loader)}")
    outputs = test(model, test_loader, device, thing_to_save, where_to_save=f"{outputs_dir}/{thing_to_save}/test")
    if outputs is not None: Utils.save_as_pickle(outputs, f"{outputs_dir}/test_{thing_to_save}.pkl")
