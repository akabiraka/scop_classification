import sys
sys.path.append("../scop_classification")
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import utils as Utils

class SCOPDataset(Dataset):
    def __init__(self, inp_file, class_dict, n_attn_heads, task="SF", max_len=1024, attn_type="contactmap") -> None:
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
        self.attn_type = attn_type #contactmap, nobackbone, longrange


    def __len__(self):
        return self.df.shape[0]

    
    def padd(self, x):
        """ x (torch.Tensor): [len, dim_embed]. The len can be different, and we will padd that dimension.
        """
        x = x[:self.max_len] #truncate to max_len
        src = torch.cat([x, torch.zeros([self.max_len-x.shape[0], x.shape[1]])]) # padding 0's at the end
        padding_mask = torch.cat([torch.zeros([x.shape[0]]), torch.ones([self.max_len-x.shape[0]])]) # non-zero values indicates ignore
        out = {"src": src.to(dtype=torch.int32),
               "key_padding_mask": padding_mask.to(dtype=torch.bool)}
        return out

    def gets_attn_mask(self, dist_matrix):
        if self.attn_type=="nobackbone": 
            contact_map = np.where((dist_matrix>1.0) & (dist_matrix<8.0), 1, 0)
        elif self.attn_type=="longrange": 
            contact_map = np.where((dist_matrix>4.0) & (dist_matrix<8.0), 1, 0)
        elif self.attn_type=="contactmap":
            contact_map = np.where(dist_matrix<8.0, 1, 0)
        else: 
            raise NotImplementedError("Unknown attn_type value passed.")

        contact_map = contact_map[:self.max_len, :self.max_len] #truncate to max_len
        attn_mask = torch.ones([self.max_len, self.max_len]) #necessary padding
        attn_mask[:contact_map.shape[0], :contact_map.shape[1]] = torch.tensor(contact_map)
        attn_mask = torch.logical_not(attn_mask.to(dtype=torch.bool))
        attn_mask = attn_mask.repeat(self.n_attn_heads, 1, 1)
        return attn_mask


    def __getitem__(self, index):
        row = self.df.loc[index]
        pdb_id, chain_and_region = row["FA-PDBID"].lower(), row["FA-PDBREG"].split(":")
        chain_id, region = chain_and_region[0], chain_and_region[1]

        feature = Utils.load_pickle("data/features/"+pdb_id+chain_id+region+".pkl")
        dist_matrix = Utils.load_pickle("data/distance_matrices/"+pdb_id+chain_id+region+".pkl")
        
        # features that will be used in the model
        feature = torch.tensor(feature[:, -20:]) #taking only 1-hot feature
        
        #int amino-acid encoding [1, 20]
        feature = torch.argmax(feature, dim=1)+1 
        feature.unsqueeze_(1)

        # computing src and key_padding_mask
        data = self.padd(feature)
        data["src"].squeeze_(1)
        
        # computing attention mask
        data["attn_mask"] = self.get_attn_mask(dist_matrix)

        # making ground-truth class tensor
        class_id = self.class_dict[self.df.loc[index, self.task]]
        label = F.one_hot(torch.tensor(class_id), len(self.class_dict)).to(dtype=torch.float32)
        
        return data, label