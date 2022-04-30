import sys
sys.path.append("../scop_classification")
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from sklearn.metrics import accuracy_score
# CUDA_LAUNCH_BLOCKING=1

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, dim_embed, dim_ff, dropout=0.3):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim_embed, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, dim_embed, dropout=0.2):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(dim_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, dim_embed, attention, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(dim_embed, dropout), 2)
        self.dim_embed = dim_embed

    def forward(self, x, key_padding_mask, attn_mask):
        attn_output, _ = self.attention(x, x, x, key_padding_mask, attn_mask=attn_mask)
        x = self.sublayer[0](x, lambda x: attn_output)
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.dim_embed)
        
    def forward(self, x, key_padding_mask, attn_mask):
        for layer in self.layers:
            x = layer(x, key_padding_mask, attn_mask)
        return self.norm(x)


class PairwiseDistanceDecoder(nn.Module):
    def __init__(self):
        super(PairwiseDistanceDecoder, self).__init__()

    def forward(self, h, sigmoid=True):
        l2_dist = torch.cdist(h, h, p=1)
        return torch.sigmoid(l2_dist) if sigmoid else l2_dist


class Classification(nn.Module):
    def __init__(self, dim_embed, n_classes, dropout=0.5):
        super(Classification, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(dim_embed, int(dim_embed/2)),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(int(dim_embed/2), n_classes))
        # do not use softmax as last layer when using cross-entropy loss
    def forward(self, x):
        """x (torch.Tensor): shape [batch_size, len, dim_embed]"""
        x = torch.mean(x, dim=1) #global average pooling. shape [batch_size, dim_embed]
        x = self.classifier(x)
        return x


class Embeddings(nn.Module):
    def __init__(self, vocab_size, dim_embed):
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(vocab_size, dim_embed, padding_idx=0)
        self.dim_embed = dim_embed

    def forward(self, x):
        return self.embed(x) * np.sqrt(self.dim_embed)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, dim_embed, dropout, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dim_embed)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_embed, 2) *
                             -(np.log(10000.0) / dim_embed))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].clone().detach().requires_grad_(False)
        return self.dropout(x)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, dim_embed, dim_ff, n_attn_heads, n_encoder_layers, n_classes, dropout, max_len, include_embed_layer=False):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dim_embed = dim_embed
        self.dim_ff = dim_ff
        self.n_attn_heads = n_attn_heads
        self.n_encoder_layers = n_encoder_layers
        self.n_classes = n_classes
        self.dropout = dropout
        self.max_len = max_len
        self.include_embed_layer = include_embed_layer
        if self.include_embed_layer:
            self.embed_layer = nn.Sequential(Embeddings(21, dim_embed), #[0, 20] inclusive
                                            PositionalEncoding(dim_embed, dropout, max_len))
    
    def forward(self, x, key_padding_mask, attn_mask):
        if self.include_embed_layer:
            x = self.embed_layer(x)
            #print(x.shape)
        x = self.encoder(x, key_padding_mask, attn_mask)
        #print(x.shape)
        x = self.decoder(x)
        #print(x.shape)
        return x



def build_model(max_len, dim_embed, dim_ff, n_attn_heads, n_encoder_layers, n_classes, dropout=0.2, include_embed_layer=False):
    cp = copy.deepcopy
    attn = nn.MultiheadAttention(dim_embed, n_attn_heads, batch_first=True)
    ff = PositionwiseFeedForward(dim_embed, dim_ff, dropout)
    enc = Encoder(EncoderLayer(dim_embed, cp(attn), cp(ff), dropout), n_encoder_layers)
    classifier = Classification(dim_embed, n_classes, dropout) # dec = PairwiseDistanceDecoder()
    model = EncoderDecoder(enc, classifier, dim_embed, dim_ff, n_attn_heads, n_encoder_layers, n_classes, dropout, max_len, include_embed_layer)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

def count_parameters(model):
    trainable_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_weights


def padd(input:list, max_len:int):
    """
    Args:
        input (_type_): list of [len, dim_embed]. 
        The len can be different, and we will padd that dimension.
        max_len (_type_): int
    """
    input = [x[:max_len] for x in input] #truncate to max_len
    src = [torch.cat([x, torch.zeros([max_len-x.shape[0], x.shape[1]])]).unsqueeze(0) for x in input] # padding 0's at the end
    padding_mask = [torch.cat([torch.zeros([x.shape[0]]), torch.ones([max_len-x.shape[0]])]).unsqueeze(0) for x in input] # non-zero values indicates ignore
    out = {"src": torch.cat(src),
           "src_key_padding_mask": torch.cat(padding_mask)}
    return out

def compute_accuracy(y_true, y_pred):
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = np.argmax(y_pred, axis=1)
    # print(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return acc

def train(model, optimizer, criterion, train_loader, device):
    model.train()
    losses = []
    for i, (data, y_true) in enumerate(train_loader):
        x, key_padding_mask, attn_mask = data["src"].to(device), data["key_padding_mask"].to(device), data["attn_mask"].to(device)
        attn_mask = torch.cat([i for i in attn_mask])
        # print(x.shape, key_padding_mask.shape, attn_mask.shape)
        model.zero_grad()
        y_pred = model(x, key_padding_mask, attn_mask)
        loss = criterion(y_pred, y_true.to(device))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"    train batch: {i}, loss: {loss.item()}")
        # break
    return np.mean(losses)


def test(model, criterion, loader, device):
    model.eval()
    losses, acc_list = [], []
    with torch.no_grad():
        for i, (data, y_true) in enumerate(loader):
            x, key_padding_mask, attn_mask = data["src"].to(device), data["key_padding_mask"].to(device), data["attn_mask"].to(device)
            attn_mask = torch.cat([i for i in attn_mask])
            model.zero_grad()
            y_pred = model(x, key_padding_mask, attn_mask)
            loss = criterion(y_pred, y_true.to(device))
            losses.append(loss.item())
            print(f"    test batch: {i}, loss: {loss.item()}")
            
            acc = compute_accuracy(y_true, y_pred)
            acc_list.append(acc)
            print(f"                    acc: {acc}")

    return np.mean(losses), np.mean(acc_list)


# if __name__ == "__main__":
#     # hyperparameters
#     max_len = 3
#     dim_embed=3
#     n_attn_heads=1 #dim_embed must be divisible by num_head
#     dim_ff=12
#     n_encoder_layers=2
#     n_classes = 5
#     dropout=0.2

#     # dummy input
#     src1 = torch.randn([1, 3]) #will be padded
#     src2 = torch.randn([2, 3]) #will be padded
#     src3 = torch.randn([3, 3]) #will not be padded
#     src4 = torch.randn([4, 3]) #will be truncated
#     x = padd([src1, src2, src3, src4], max_len)
#     src, key_padding_mask = x["src"], x["src_key_padding_mask"].to(dtype=torch.bool)
#     # the rows corresponding to padding, must have the attention mask 0. 
#     attn_mask = torch.tensor([[[0, 1, 1], [0, 0, 0], [0, 0, 0]],
#                             [[0, 0, 1], [0, 0, 1], [0, 0, 0]],
#                             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
#                             [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=torch.bool)
#     attn_mask = attn_mask.repeat(n_attn_heads, 1, 1)
#     print(src.shape, key_padding_mask, attn_mask.shape)

    
#     # test torch multihead attention
#     multihead_attn = nn.MultiheadAttention(dim_embed, n_attn_heads, batch_first=True)
#     attn_output, attn_output_weights = multihead_attn(src, src, src, key_padding_mask, attn_mask=attn_mask)
#     print(attn_output)
#     print(attn_output_weights)

#     # build model and run
#     model = BuildModel(dim_embed, dim_ff, n_attn_heads, n_encoder_layers, n_classes, dropout)
#     out = model(src, key_padding_mask, attn_mask)
#     print(out.shape, out)
