import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, loc_num, embedding_dim, hidden_dim, latent_dim, n_layers=2):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(loc_num, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.gru(embedded, None)
        output = self.out(output)
        return output
        
        
class Decoder(nn.Module):
    def __init__(self, loc_num, embedding_dim, hidden_dim, latent_dim, n_layers=2):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(loc_num, embedding_dim)
        self.gru = nn.GRU(embedding_dim + latent_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, loc_num - 2)
        
    def forward(self, latent_code, x, length=54):
        latent_code = latent_code.repeat(length, 1, 1).permute(1, 0, 2)
        embedded = self.embedding(x)
        gru_in = torch.cat([latent_code, embedded], dim=-1)
        output, _ = self.gru(gru_in, None)
        return self.out(output)