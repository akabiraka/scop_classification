import sys
sys.path.append("../scop_classification")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.SCOPDataset import SCOPDataset

data_file_path="data/splits/debug/all_cleaned.txt"
task="SF"
df = pd.read_csv(data_file_path)
x = df[task].unique().tolist()
class_dict = {j:i for i,j in enumerate(x)}
scop = SCOPDataset(inp_file=data_file_path, class_dict=class_dict, n_attn_heads=1, task=task, max_len=150, attn_type="contactmap")
print(len(scop))
data, label = scop.__getitem__(0)
print(data["src"].shape, data["key_padding_mask"].shape, data["attn_mask"].shape, label.shape)
print(f"class label: {label}")

img_format = "png"


# plotting a sequence int embedding
src = data["src"].unsqueeze(0).numpy()
src = np.repeat(src, 10, axis=0)
plt.imshow(src, interpolation='none')
plt.yticks([])
# plt.show()
plt.savefig(f"outputs/images/input_sequence.{img_format}", dpi=300, format=img_format, bbox_inches='tight', pad_inches=0.0)

# plotting key padding mask
key_padding_mask = data["key_padding_mask"].unsqueeze(0).numpy()
key_padding_mask = np.repeat(key_padding_mask, 10, axis=0)
plt.imshow(key_padding_mask, interpolation='none')
plt.yticks([])
# plt.show()
plt.savefig(f"outputs/images/key_padding_mask.{img_format}", dpi=300, format=img_format, bbox_inches='tight', pad_inches=0.0)


# plotting attention mask
# print(data["attn_mask"].squeeze(0).numpy())
plt.cla()
plt.imshow(data["attn_mask"].squeeze(0).numpy(), interpolation='none')
# plt.show()
plt.savefig(f"outputs/images/attn_mask.{img_format}", dpi=300, format=img_format, bbox_inches='tight', pad_inches=0.0)

