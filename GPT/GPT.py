import torch
import numpy as np
import torch.nn as nn
import numpy as np
import math
import pandas as pd 
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset





class Embedding(nn.Module):

    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
            
    def forward(self, x):
        return self.embedding(x) 
    


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create a matrix of [max_len, d_model] representing positions
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate the division term using the log-space trick for numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Fill the even indices with sine and odd with cosine
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # register_buffer ensures pe is moved to GPU with the model but not trained
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # Add the positional encoding to the embeddings
        x = x + self.pe[:, :x.size(1), :]
        return x


class multihead_attention(torch.nn.Module):
    
    def __init__(self, d_model, num_heads):
        super(multihead_attention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)
    

    
    def forward(self, x):
        batch_size, seq_length, d_model = x.size()
        
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, d_model)
        
        output = self.W_o(attn_output)
        
        return output 
    

class layernormalization(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(layernormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
       
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized_x = (x - mean) / (std + self.eps)
        output = self.gamma * normalized_x + self.beta
        return output


class feedforward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(feedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        output = self.linear2(x)
        return output