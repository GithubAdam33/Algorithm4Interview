# package: transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
import torch.optim as optim
import torch.utils.data as data

#Multihead attention
class MultiHeadedAttention(nn.Module):
    def __init__(self,d_model,n_head):
        super.__init__()
        self.d_model = d_model
        self.n_head = n_head
        assert(self.d_model % self.n_head == 0)
        self.d_k = self.d_model // self.n_head
        
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_o = nn.Linear(d_model,d_model)
    
    def scaled_attention(self, q, k, v, mask = None):
        attention = torch.matmul(q, k.tanspose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        attention_probs = torch.softmax(attention, dim = -1)
        output = torch.matmul(attention_probs,v)
        return output
    
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.n_head, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forwward(self, q, k, v, mask = None):
        q = self.split_heads(self.w_q(q)) #b,h,s,d
        k = self.split_heads(self.w_k(k))
        v = self.split_heads(self.w_v(v))
        atten_output = self.scaled_attention(q,k,v,mask)
        output = self.w_o(self.combine_heads(atten_output)) #b,s,q
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
    
class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__( )

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[: ,1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]        
        
class EncoderLayer(nn.modules):
    def __init__(self, d_model, nums_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadedAttention(d_model, nums_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nums_heads, d_ff, dropout):
        super().__init__()
        self.self_attn1 = MultiHeadedAttention(d_model, nums_heads)
        self.self_attn2 = MultiHeadedAttention(d_model, nums_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, encoder_output, x, mask_encoder, mask_x):
        attn1_out = self.self_attn1(x, x, x, mask_x)
        x = self.norm1(x + self.dropout(attn1_out))
        attn2_out = self.self_attn2(x, encoder_output, encoder_output, mask_encoder)
        x = self.norm2(x + self.dropout(attn2_out))
        ff_output = self.feed_forward(x)
        output = self.norm3(x + self.dropout(ff_output))
        return output
    
class Transformer(nn.modules):
    def __init__(self, src_size, tgt_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super.__init__()
        self.encoder_embedding = nn.Embedding(src_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_size, d_model)
        self.positional_encoding = PositionEncoding(d_model, max_seq_length)
        
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        self.fc = nn.Linear(d_model, tgt_size)
        self.dropout = nn.Dropout(dropout)
        
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src, tgt = self.encoder_embedding(src), self.decoder_embedding(tgt)
        src, tgt = self.positional_encoding(src), self.positional_encoding(tgt)
        encoder_out = src
        for enc_layer in self.encoder_layers:
            encoder_out = enc_layer(encoder_out, src_mask)
        decoder_out = tgt
        for dec_layer in self.decoder_layers:
            decoder_out = dec_layer(enc_layer,decoder_out,src_mask,tgt_mask)
            
        output = self.fc(decoder_out)
        return output
        
