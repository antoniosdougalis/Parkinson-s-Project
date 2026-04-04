#%% Multi head Attention Transformer Arhitecture for EEG Feature Classification

# Part of the Submitted Manuscript 
# Dougalis Antonios 2026, Interpretable Electrophysiological Features of Resting-State EEG Reveal Distributed Cortical Signatures in Parkinson’s Disease, arXiv;

'''
#%% Multi head Attention Transformer for EEG Feature Classifiecation

written by Antonios Dougalis, Feb 2026, Kuopio Finland
contact: antoniosdougalis (at) gmail.com

'''

# Import libraries
import torch
import torch.nn as nn
import numpy as np

#%% Generate some synthetic data to have for inputs

# Generate synthetic data
def generate_synthetic_data(num_samples, input_dim, seq_len, num_classes):
    
    # Random EEG data (num_samples, input_dim, seq_len )
    X = np.random.rand(num_samples, input_dim, seq_len).astype(np.float32)  # 10 time steps
    
    # Random labels
    y = np.random.randint(0, num_classes, size=(num_samples,))
    return X, y


# Create a PyTorch dataset and dataloader
class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()         
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    
#%% MultiHeadAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, printToggle=False):
        super(MultiHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        self.printToggle = printToggle
        
        # the Query, Key and Value and the Combined Final Dense layer
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size): # (batch_size, sequence_length, d_model)
        
        x = x.view(batch_size, -1, self.num_heads, self.depth) # (batch_size, seq_len, num_heads, depth).
        x = x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, depth)
        return x

    def forward(self, x):
        # (input shape: (batch_size, seq_len, d_model))
        if self.printToggle: print(f' shape of data INPUT {x.shape} ') 
        batch_size = x.size(0)
        
        # Calculate the k, q, v parameters from the x input 
        q = self.split_heads(self.wq(x), batch_size) # (batch_size, num_heads, seq_len, depth)
        k = self.split_heads(self.wk(x), batch_size)
        v = self.split_heads(self.wv(x), batch_size)
        
        if self.printToggle: print(f' shape of k, q, v {k.shape} \n representing (batch_size, num_heads, seq_len, depth) ')
        
        # Scaled dot-product attention: # This represents the attention scores for each head.
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))  #  (batch_size, num_heads, in_chans, in_chans)
        if self.printToggle: print(f' shape of matmul_qk {matmul_qk.shape} is (batch_size, num_heads, seq_len, seq_len)')
                                   
        # Scale the scores
        dk = torch.tensor(self.depth, dtype=torch.float32)  # Scale by the depth
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)
        if self.printToggle: print(f' shape of scaled_attention_logits {scaled_attention_logits.shape} is (batch_size, num_heads, seq_len, seq_len)')

        # Apply mask if provided (optional)
        # if mask is not None:
            # apply mask for future tokens
            # pastmask = torch.tril(torch.ones(n_batch,context_length,context_length))
            # qk_scaled[pastmask==0] = -torch.inf # equivalent to adding a matrix of zeros/-infs

        # Softmax to get attention weights
        attention_weights = torch.softmax(scaled_attention_logits, dim=-1)  
        if self.printToggle: print(f' shape of attention_weights {attention_weights.shape} is (batch_size, num_heads, seq_len, seq_len)')
        
        # Multiply attention weights by values
        output = torch.matmul(attention_weights, v)  
        if self.printToggle: print(f' shape of output= attention_weights X weights {output.shape} is (batch_size, num_heads, seq_len, depth)')
        
        # Concatenate heads and put through final dense layer
        output = output.permute(0, 2, 1, 3).contiguous()
        if self.printToggle: print(f' shape of permuted & Conc Output {output.shape} is (batch_size, seq_len, num_heads,  depth)')
        
        # reconstruct the ebedding dim (d_model)
        output = output.view(batch_size, -1, self.d_model)  
        
        output = self.dense(output)  # Final linear layer: LInear Transfomration!
        # output = output*self.dense(output)  # Final linear layer: Choose for Non linear tranfmromation!
        
        if self.printToggle: print(f' shape of data OUTPUT MUST match INPUT {output.shape} representing (batch_size, seq_len, d_model) ')

        return output

#%% FeedForward

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout, printToggle=False):
        
        # d_model: The size of the input features (e.g., 512).
        # d_ff: The size of the feedforward hidden layer (e.g., 2048).
        
        super(FeedForward, self).__init__()
        self.printToggle = printToggle
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # input x and output x will be (batch_size, in_chans, seq_len) 
        x = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        
        if self.printToggle: print(f' data after FeedForward OUTPUT {x.shape} representing (batch_size, in_chans, seq_len) ')
        return x

#%% EncoderLayer

class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, num_heads, d_ff, dropout, printToggle=False ):
        super(EncoderLayer, self).__init__()
        
        self.printToggle = printToggle
        self.mha = MultiHeadAttention(d_model, num_heads, False)
        self.ffn = FeedForward(d_model, d_ff, dropout, False)
        self.AttNorm = nn.LayerNorm(d_model)
        self.MLPNnorm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        
        # enter attention block
        res = x
        attn_output = self.mha( self.AttNorm(x) )
        x = res + self.dropout1(attn_output)
        
        # enter MLP block
        res = x
        ffn_output = self.ffn(self.MLPNnorm(x))
        x = res + self.dropout2(ffn_output)
        
        if self.printToggle: print(f' data Encoder Layer OUTPUT {x.shape} representing (batch_size, seq_len, d_model) ')

        return x

#%% TransformerEncoder

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_dim, dropout, printToggle=False ):
        super(TransformerEncoder, self).__init__()
        
        self.printToggle = printToggle
        self.embedding = nn.Linear(input_dim, d_model)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout, False) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.printToggle: print(f' data INPUT {x.shape} representing (batch_size, in_dims, seq_len) ')
        
        # embed the data f
        x = self.embedding(x)
        if self.printToggle: print(f' data after Embedding {x.shape} representing (batch_size, seq_len, d_model) ')
        
        # Apply dropout AFTER embedding
        # x = self.dropout(x)
                
        # pass embedding to One or Several multihead attention MLP transformer layers
        for layer in self.enc_layers:
            x = layer(x)
        if self.printToggle: print(f' data after Transformer Encoder OUTPUT {x.shape} representing (batch_size, seq_len, d_model) ')
    
        return x
    

#%% EmT Module

class EmT(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_dim, num_classes, dropout, printToggle = False):
        super(EmT, self).__init__()
        self.printToggle = printToggle
        self.LayerNorm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, input_dim, dropout)
        self.final_layer = nn.Linear(d_model, num_classes)
                

    def forward(self, x):
        
        x = self.encoder(x)
        
        # x = x[:, -1, :]  # Get the last output for classification
        x = x.mean(dim=1)  # Get the mean of the channels as the output
        
        x = self.LayerNorm(x)
        x = self.final_layer(x)
        if self.printToggle: print(f' Final Data OUTPUT {x.shape} representing (batch_size, num_classes ')
        x = x.squeeze()
        return x 


