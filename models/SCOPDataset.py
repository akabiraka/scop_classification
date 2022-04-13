from asyncio import tasks
import sys
from numpy import dtype
sys.path.append("../scop_classification")
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import utils as Utils

class SCOPDataset(Dataset):
    def __init__(self, inp_file, class_dict, n_attn_heads, task="SF", max_len=1024) -> None:
        """Creates SCOP dataset.
        Args:
            inp_file (str): train/val/test file path
            task (str, optional): Any SCOP classification tasks, i.e: TP, CL, CF, SF, FA
            max_len (int, optional): Defaults to 1024.
        """
        super(SCOPDataset, self).__init__()
        self.df = pd.read_csv(inp_file)
        
        self.class_dict = class_dict
        self.task = task
        self.max_len = max_len
        self.n_attn_heads = n_attn_heads


    def __len__(self):
        return self.df.shape[0]

    
    def padd(self, x):
        """ x (torch.Tensor): [len, dim_embed]. The len can be different, and we will padd that dimension.
        """
        x = x[:self.max_len] #truncate to max_len
        src = torch.cat([x, torch.zeros([self.max_len-x.shape[0], x.shape[1]])]) # padding 0's at the end
        padding_mask = torch.cat([torch.zeros([x.shape[0]]), torch.ones([self.max_len-x.shape[0]])]) # non-zero values indicates ignore
        out = {"src": src,
               "key_padding_mask": padding_mask.to(dtype=torch.bool)}
        return out

    def get_attn_mask(self, contact_map):
        attn_mask = torch.ones([self.max_len, self.max_len])
        attn_mask[:contact_map.shape[0], :contact_map.shape[1]] = torch.tensor(contact_map)
        attn_mask = torch.logical_not(attn_mask.to(dtype=torch.bool))
        attn_mask = attn_mask.repeat(self.n_attn_heads, 1, 1)
        return attn_mask


    def __getitem__(self, index):
        row = self.df.loc[index]
        pdb_id, chain_and_region = row["FA-PDBID"].lower(), row["FA-PDBREG"].split(":")
        chain_id, region = chain_and_region[0], chain_and_region[1]

        feature = Utils.load_pickle("data/features/"+pdb_id+chain_id+region+".pkl")
        contact_map = Utils.load_pickle("data/contact_maps/"+pdb_id+chain_id+region+".pkl")

        # getting src and key_padding_mask
        data = self.padd(torch.tensor(feature, dtype=torch.float32))
        
        # getting attention mask
        data["attn_mask"] = self.get_attn_mask(contact_map)

        # making ground-truth class tensor
        class_id = self.class_dict[self.df.loc[index, self.task]]
        label = F.one_hot(torch.tensor(class_id), len(self.class_dict)).to(dtype=torch.float32)
        
        return data, label
        

# scop = SCOPDataset(inp_file="data/splits/train_7.txt", n_classes=3, task="FA", max_len=512)
# print(len(scop))
# data, label = scop.__getitem__(5)
# print(data["src"].shape, data["key_padding_mask"].shape, data["attn_mask"].shape, label.shape)
# print(data["attn_mask"])