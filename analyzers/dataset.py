import sys
sys.path.append("../scop_classification")
import pandas as pd
import matplotlib.pyplot as plt
from models.SCOPDataset import SCOPDataset

df = pd.read_csv("data/splits/all_10.txt")
x = df["SF"].unique().tolist()
class_dict = {j:i for i,j in enumerate(x)}
scop = SCOPDataset(inp_file="data/splits/all_10.txt", class_dict=class_dict, n_attn_heads=1, task="SF", max_len=150)
print(len(scop))
data, label = scop.__getitem__(4)
print(data["src"].shape, data["key_padding_mask"].shape, data["attn_mask"].shape, label.shape)


# plotting attention mask
# print(data["attn_mask"].squeeze(0).numpy())
plt.imshow(data["attn_mask"].squeeze(0).numpy(), interpolation='none')
# plt.show()
plt.savefig(f"outputs/images/example_attn_mask.png", dpi=300, format="png", bbox_inches='tight', pad_inches=0.0)

# plotting key padding mask
# print(data["key_padding_mask"].numpy())
plt.imshow(data["key_padding_mask"].unsqueeze(0).numpy(), interpolation='none')
plt.yticks([])
# plt.show()
plt.savefig(f"outputs/images/example_key_padding_mask.png", dpi=300, format="png", bbox_inches='tight', pad_inches=0.0)