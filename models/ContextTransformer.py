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

    def forward(self, x, sublayer, sublayer_no=None, key_padding_mask=None, attn_mask=None):
        attn_weights = None
        if sublayer_no==0:
            attn_output, attn_weights = sublayer(self.norm(x))
            x = x + self.dropout(attn_output)
        else: 
            x = x + self.dropout(sublayer(self.norm(x)))
        return x, attn_weights


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
        # y = self.norm(x)
        # attn_output, attn_weights = self.attention(y, y, y, key_padding_mask, attn_mask)
        # x = x + self.dropout(attn_output)

        # return self.sublayer[0](x, self.feed_forward)

        # replace the next line with above 3 lines to separate the attention-outputs and weights
        x, attn_weights = self.sublayer[0](x, lambda y: self.attention(y, y, y, key_padding_mask, attn_mask), 0, key_padding_mask, attn_mask)
        x, _ = self.sublayer[1](x, self.feed_forward, 1)
        return x, attn_weights


class Encoder(nn.Module):
    def __init__(self, layer, N, return_attn_weights=True):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.dim_embed)
        self.return_attn_weights = return_attn_weights
        
    def forward(self, x, key_padding_mask, attn_mask):

        # do not keep and return the attention weights
        if not self.return_attn_weights:
            for i, layer in enumerate(self.layers):
                x, attn_weights = layer(x, key_padding_mask, attn_mask)
            return self.norm(x), None 
            # None b.c. of storing such amount of data does not make sense 
        
        # store and return the attention weights for all layers and heads
        else:
            all_layers_attn_weights = []
            for i, layer in enumerate(self.layers):
                x, attn_weights = layer(x, key_padding_mask, attn_mask)
                all_layers_attn_weights.append(attn_weights)
            
            all_layers_attn_weights = torch.stack(all_layers_attn_weights, dim=0)
            # print(all_layers_attn_weights.shape) #[n_layers, batch_size, n_heads, max_len, max_len]
            return self.norm(x), all_layers_attn_weights


class PairwiseDistanceDecoder(nn.Module):
    def __init__(self):
        super(PairwiseDistanceDecoder, self).__init__()

    def forward(self, h, sigmoid=True):
        l2_dist = torch.cdist(h, h, p=1)
        return torch.sigmoid(l2_dist) if sigmoid else l2_dist


class Classification(nn.Module):
    def __init__(self, dim_embed, n_classes, dropout=0.5):
        super(Classification, self).__init__()

        self.attn_linear = torch.nn.Linear(dim_embed, 1)
        self.classifier = nn.Linear(dim_embed, n_classes)

    def forward(self, last_hidden_state):
        """last_hidden_state (torch.Tensor): shape [batch_size, seq_len, dim_embed]"""
        # x = torch.mean(x, dim=1) #global average pooling. shape [batch_size, dim_embed]
        activation = torch.tanh(last_hidden_state) # [batch_size, seq_len, dim_embed]

        score = self.attn_linear(activation) # [batch_size, seq_len, 1]      
        weights = torch.softmax(score, dim=1) # [batch_size, seq_len, 1]
        last_layer_learned_rep = torch.sum(weights * last_hidden_state, dim=1)  # [batch_size, dim_embed]

        cls_pred = self.classifier(last_layer_learned_rep) # [batch_size, n_classes]
        return cls_pred, last_layer_learned_rep


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
        x, _ = self.encoder(x, key_padding_mask, attn_mask)
        #print(x.shape)
        cls_pred, _ = self.decoder(x)
        #print(x.shape)
        return cls_pred
    
    def get_embeddings(self, x):
        return self.embed_layer(x)
    
    def get_all_layers_attn_weights(self, x, key_padding_mask, attn_mask):
        """This also returns embeddings to avaid extra computation."""
        embeddings = self.embed_layer(x)
        _, all_layers_attn_weights = self.encoder(embeddings, key_padding_mask, attn_mask)
        return all_layers_attn_weights, embeddings

    def get_last_layer_learned_rep(self, x, key_padding_mask, attn_mask):
        """This also returns all_layers_attn_weights and embeddings to avaid extra computation."""
        embeddings = self.embed_layer(x)
        x, all_layers_attn_weights = self.encoder(embeddings, key_padding_mask, attn_mask)
        _, last_layer_learned_rep = self.decoder(x)
        return last_layer_learned_rep, all_layers_attn_weights, embeddings

    def get_all(self, x, key_padding_mask, attn_mask):
        embeddings = self.embed_layer(x)
        x, all_layers_attn_weights = self.encoder(embeddings, key_padding_mask, attn_mask)
        y_pred, last_layer_learned_rep = self.decoder(x)
        return y_pred, last_layer_learned_rep, all_layers_attn_weights, embeddings


class MultiheadAttentionWrapper(nn.Module):
    def __init__(self, dim_embed, n_attn_heads, batch_first=True, apply_attn_mask=True, apply_neighbor_aggregation=False) -> None:
        super(MultiheadAttentionWrapper, self).__init__()
        self.attn = nn.MultiheadAttention(dim_embed, n_attn_heads, batch_first=batch_first)
        self.n_attn_heads = n_attn_heads
        self.apply_attn_mask = apply_attn_mask
        self.apply_neighbor_aggregation = apply_neighbor_aggregation
    
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        if self.apply_attn_mask and self.apply_neighbor_aggregation:
            attn_output, attn_weights = self.attn(query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask, average_attn_weights=False)
            #print(attn_output.dtype, attn_weights.dtype, attn_mask[range(0, attn_mask.shape[0], self.n_attn_heads), :, :].to(dtype=torch.float32).dtype)
            attn_output = torch.matmul(attn_mask[range(0, attn_mask.shape[0], self.n_attn_heads), :, :].to(dtype=torch.float32), attn_output) #neighborhood aggregation
        
        elif self.apply_attn_mask:
            attn_output, attn_weights = self.attn(query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask, average_attn_weights=False)
            # print(attn_output.shape, attn_weights.shape)
        
        else: #Case: no attention mask
            attn_output, attn_weights = self.attn(query, key, value, key_padding_mask=key_padding_mask, average_attn_weights=False)
            
        return attn_output, attn_weights


def build_model(max_len, dim_embed, dim_ff, n_attn_heads, n_encoder_layers, n_classes, dropout=0.2, 
                include_embed_layer=False, apply_attn_mask=True, apply_neighbor_aggregation=False, return_attn_weights=False):
    cp = copy.deepcopy
    attn = MultiheadAttentionWrapper(dim_embed, n_attn_heads, batch_first=True, apply_attn_mask=apply_attn_mask, apply_neighbor_aggregation=apply_neighbor_aggregation)
    ff = PositionwiseFeedForward(dim_embed, dim_ff, dropout)
    enc = Encoder(EncoderLayer(dim_embed, cp(attn), cp(ff), dropout), n_encoder_layers, return_attn_weights)
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


def train(model, optimizer, criterion, train_loader, device):
    model.train()
    losses = []
    for i, (data, y_true) in enumerate(train_loader):
        x, key_padding_mask, attn_mask = data["src"].to(device), data["key_padding_mask"].to(device), data["attn_mask"].to(device)
        attn_mask = torch.cat([i for i in attn_mask])
        # print(x.shape, key_padding_mask.shape, attn_mask.shape)
        model.zero_grad(set_to_none=True)
        y_pred = model(x, key_padding_mask, attn_mask)
        loss = criterion(y_pred, y_true.to(device))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"    train batch: {i}, loss: {loss.item()}")

        # acc = compute_accuracy(y_true, y_pred)
        # print(f"                    acc: {acc}")
        # break
    return np.mean(losses)


@torch.no_grad()
def test(model, criterion, loader, device):
    model.eval()
    losses, pred_labels, true_labels = [], [], []
    for i, (data, y_true) in enumerate(loader):
        x, key_padding_mask, attn_mask = data["src"].to(device), data["key_padding_mask"].to(device), data["attn_mask"].to(device)
        attn_mask = torch.cat([i for i in attn_mask])
        model.zero_grad(set_to_none=True)
        y_pred = model(x, key_padding_mask, attn_mask)
        loss = criterion(y_pred, y_true.to(device))
        
        losses.append(loss.item())
        pred_labels.append(y_pred.argmax(dim=1).cpu().numpy())
        true_labels.append(y_true.cpu().numpy())
        
    metrics = get_metrics(true_labels, pred_labels)
    loss = np.mean(losses)
    return loss, metrics


def get_metrics(target_classes, pred_classes):
    from sklearn.metrics import accuracy_score, recall_score, precision_score
    acc = accuracy_score(target_classes, pred_classes)
    precision = precision_score(target_classes, pred_classes, average="weighted", zero_division=1)
    recall = recall_score(target_classes, pred_classes, average="weighted",  zero_division=1)
    return {"acc": acc, 
            "precision": precision, 
            "recall": recall, 
            "pred_classes": pred_classes, 
            "target_classes": target_classes}

# def compute_accuracy(y_true, y_pred):
#     y_true = y_true.cpu().detach().numpy()
#     y_pred = y_pred.cpu().detach().numpy()
#     y_pred = np.argmax(y_pred, axis=1)
#     # print(y_true, y_pred)
#     acc = accuracy_score(y_true, y_pred)
#     return acc

# @torch.no_grad()
# def test(model, criterion, loader, device):
#     model.eval()
#     losses, acc_list = [], []
#     for i, (data, y_true) in enumerate(loader):
#         x, key_padding_mask, attn_mask = data["src"].to(device), data["key_padding_mask"].to(device), data["attn_mask"].to(device)
#         attn_mask = torch.cat([i for i in attn_mask])
#         model.zero_grad(set_to_none=True)
#         y_pred = model(x, key_padding_mask, attn_mask)
#         loss = criterion(y_pred, y_true.to(device))
#         losses.append(loss.item())
#         # print(f"    test batch: {i}, loss: {loss.item()}")
        
#         acc = compute_accuracy(y_true, y_pred)
#         acc_list.append(acc)
#         # print(f"                    acc: {acc}")

#     return np.mean(losses), np.mean(acc_list)


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
