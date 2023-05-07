import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np
import scipy.sparse as sp

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
 
# +------------------------------------------------+
# | Temporal Multiresolution Graph Neural Networks |
# +------------------------------------------------+

class TMGNN(nn.Module):
    def __init__(self, nfeat, nhid, nout, n_nodes, window, dropout, num_clusters = [10, 5], use_norm = False):
        super(TMGNN, self).__init__()
        self.window = window
        self.n_nodes = n_nodes
        self.nhid = nhid
        self.nfeat = nfeat
        self.use_norm = use_norm

        # +--------------------------------------+
        # | Multiresolution Graph Networks (MGN) |
        # +--------------------------------------+

        # Bottom encoder
        self.bottom_encoder = MPNN_Encoder(nfeat, nhid, nhid, dropout)

        # Number of clusters
        self.num_clusters = num_clusters

        # Multiresolution construction
        self.middle_linear = nn.ModuleList()
        self.middle_encoder = nn.ModuleList()

        for size in self.num_clusters:
            self.middle_linear.append(nn.Linear(nhid, size))
            self.middle_encoder.append(nn.Linear(nhid, nhid))

        # Mixing multiple resolutions together
        self.mix_1 = nn.Linear((len(self.num_clusters) + 1) * nhid, 512)
        self.mix_2 = nn.Linear(512, (len(self.num_clusters) + 1) * nhid)

        # +------------------------+
        # | Long Short-Term Memory |
        # +------------------------+

        self.rnn1 = nn.LSTM((len(self.num_clusters) + 1) * nhid, nhid, 1)
        self.rnn2 = nn.LSTM(nhid, nhid, 1)
        
        self.fc1 = nn.Linear(2 * nhid + window * nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nout)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, adj, x):
        lst = list()
        skip = x.view(-1, self.window, self.n_nodes, self.nfeat)
        skip = torch.transpose(skip, 1, 2).reshape(-1, self.window, self.nfeat)
        
        # +--------------------------------------+
        # | Multiresolution Graph Networks (MGN) |
        # +--------------------------------------+
        x = x.view(-1, self.nfeat)

        # All latents
        all_latents = []

        # Bottom encoder
        bottom_latent = self.bottom_encoder(adj, x)
        all_latents.append(bottom_latent)

        # Product of all assignment matrices
        product = None

        # Multiresolution construction
        adj = adj.to_dense()
        latent = bottom_latent

        for level in range(len(self.num_clusters)):
            size = self.num_clusters[level]

            # Assignment matrix
            assign = self.middle_linear[level](latent)
            assign = F.gumbel_softmax(assign, tau = 1, hard = True, dim = 1)

            # Update product
            if level == 0:
                product = assign
            else:
                product = torch.matmul(product, assign)

            # Coarsen node features
            x = torch.matmul(assign.transpose(0, 1), latent)
            x = F.normalize(x, dim = 1)

            # Coarsen the adjacency
            adj = torch.matmul(torch.matmul(assign.transpose(0, 1), adj), assign)
            adj = adj / torch.sum(adj)

            # New latent by graph convolution
            latent = torch.tanh(self.middle_encoder[level](torch.matmul(adj, x)))

            # Extended latent
            extended_latent = torch.matmul(product, latent)
            all_latents.append(extended_latent)

        # Normalization
        if self.use_norm == True:
            for idx in range(len(all_latents)):
                all_latents[idx] = all_latents[idx] / torch.norm(all_latents[idx], p = 2)

        # Concatenate all resolutions
        representation = torch.cat(all_latents, dim = 1)
        x = representation

        # Mixing multiple resolutions
        x = torch.relu(self.mix_1(x))
        x = torch.relu(self.mix_2(x))

        # +----------------------------------------------------------+
        # | Long Short-Term Memory exactly the same as other authors |
        # +----------------------------------------------------------+
        x = x.view(-1, self.window, self.n_nodes, x.size(1))
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(self.window, -1, x.size(3))#self.batch_size*self.n_nodes

        x, (hn1, cn1) = self.rnn1(x)        
        out2, (hn2,  cn2) = self.rnn2(x)

        x = torch.cat([hn1[0,:,:],hn2[0,:,:]], dim=1)
        skip = skip.reshape(skip.size(0),-1)
                
        x = torch.cat([x,skip], dim=1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x)).squeeze()
        x = x.view(-1)

        return x


# +---------------------------------------+
# | Multiresolution Graph Neural Networks |
# +---------------------------------------+

class MGNN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, num_clusters = [10, 5], use_norm = False):
        super(MGNN, self).__init__()
       
        # Use normalization or not
        self.use_norm = use_norm

        # Bottom encoder
        self.bottom_encoder = MPNN_Encoder(nfeat, nhid, nhid, dropout)
        
        # Number of clusters
        self.num_clusters = num_clusters

        # Multiresolution construction
        self.middle_linear = nn.ModuleList()
        self.middle_encoder = nn.ModuleList()

        for size in self.num_clusters:
            self.middle_linear.append(nn.Linear(nhid, size))
            self.middle_encoder.append(nn.Linear(nhid, nhid))

        # Top layer
        self.fc_1 = nn.Linear((len(self.num_clusters) + 1) * nhid, 512)
        self.fc_2 = nn.Linear(512, nout)

    def forward(self, adj, x):
        # All latents
        all_latents = []

        # Bottom encoder
        bottom_latent = self.bottom_encoder(adj, x)
        all_latents.append(bottom_latent)

        # Product of all assignment matrices
        product = None

        # Multiresolution construction
        adj = adj.to_dense()
        latent = bottom_latent

        for level in range(len(self.num_clusters)):
            size = self.num_clusters[level]
            
            # Assignment matrix
            assign = self.middle_linear[level](latent)
            assign = F.gumbel_softmax(assign, tau = 1, hard = True, dim = 1)

            # Update product
            if level == 0:
                product = assign
            else:
                product = torch.matmul(product, assign)

            # Coarsen node features
            x = torch.matmul(assign.transpose(0, 1), latent)
            x = F.normalize(x, dim = 1)

            # Coarsen the adjacency
            adj = torch.matmul(torch.matmul(assign.transpose(0, 1), adj), assign)
            adj = adj / torch.sum(adj)

            # New latent by graph convolution
            latent = torch.tanh(self.middle_encoder[level](torch.matmul(adj, x)))
            
            # Extended latent
            extended_latent = torch.matmul(product, latent)
            all_latents.append(extended_latent)

        # Normalization
        if self.use_norm == True:
            for idx in range(len(all_latents)):
                all_latents[idx] = all_latents[idx] / torch.norm(all_latents[idx], p = 2)

        # Concatenate all resolutions
        representation = torch.cat(all_latents, dim = 1)

        # Prediction
        predict = torch.relu(self.fc_2(torch.relu(self.fc_1(representation)))).squeeze().view(-1)

        # Use multiresolutioni for prediction
        return predict

class MPNN_Encoder(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(MPNN_Encoder, self).__init__()
        self.nhid = nhid

        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)

        self.fc1 = nn.Linear(nfeat+2*nhid, nhid )
        self.fc2 = nn.Linear(nhid, nout)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, adj, x):
        lst = list()
        weight = adj.coalesce().values()
        adj = adj.coalesce().indices()
        lst.append(x)

        x = self.relu(self.conv1(x,adj,edge_weight=weight))
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)

        x = self.relu(self.conv2(x, adj,edge_weight=weight))
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)

        x = torch.cat(lst, dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return x 


# +--------------------------------------------------------------------+
# | Attention Temporal Multiresolution Graph Neural Networks (GROUPED) |
# +--------------------------------------------------------------------+

class ATMGNN_GROUPED(nn.Module):
    def __init__(self, nfeat, nhid, nout, n_nodes, window, dropout, nhead = 1, num_clusters = [10, 5], num_group = 10, use_norm = False):
        super(ATMGNN_GROUPED, self).__init__()
        
        self.window = window
        self.nout = nout
        self.num_group = num_group
        self.n_nodes = n_nodes
        self.nhid = nhid
        self.nfeat = nfeat
        self.nhead = nhead
        self.use_norm = use_norm

        # +--------------------------------------+
        # | Multiresolution Graph Networks (MGN) |
        # +--------------------------------------+

        # Bottom encoder
        self.bottom_encoder = MPNN_Encoder(nfeat, nhid, nhid, dropout)

        # Number of clusters
        self.num_clusters = num_clusters

        # Multiresolution construction
        self.middle_linear = nn.ModuleList()
        self.middle_encoder = nn.ModuleList()

        for size in self.num_clusters:
            self.middle_linear.append(nn.Linear(nhid, size))
            self.middle_encoder.append(nn.Linear(nhid, nhid))

        # Mixing multiple resolutions together
        self.mix_1 = nn.Linear((len(self.num_clusters) + 1) * nhid, 512)
        self.mix_2 = nn.Linear(512, (len(self.num_clusters) + 1) * nhid)

        # +--------------------------+
        # | Attention/Self-Attention |
        # +--------------------------+

        self.self_attention = nn.MultiheadAttention((len(self.num_clusters) + 1) * nhid, self.nhead, dropout=dropout)
        self.linear_reduction = nn.Linear(self.window, 1)
        
        self.fc1 = nn.Linear((len(self.num_clusters) + 1) * nhid + window * nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nout*num_group)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        
    def forward(self, adj, x):
        lst = list()
        skip = x.view(-1, self.window, self.n_nodes, self.nfeat)
        skip = torch.transpose(skip, 1, 2).reshape(-1, self.window, self.nfeat)

        # +--------------------------------------+
        # | Multiresolution Graph Networks (MGN) |
        # +--------------------------------------+
        x = x.view(-1, self.nfeat)

        # All latents
        all_latents = []

        # Bottom encoder
        bottom_latent = self.bottom_encoder(adj, x)
        all_latents.append(bottom_latent)

        # Product of all assignment matrices
        product = None

        # Multiresolution construction
        adj = adj.to_dense()
        latent = bottom_latent

        for level in range(len(self.num_clusters)):
            size = self.num_clusters[level]

            # Assignment matrix
            assign = self.middle_linear[level](latent)
            assign = F.gumbel_softmax(assign, tau = 1, hard = True, dim = 1)

            # Update product
            if level == 0:
                product = assign
            else:
                product = torch.matmul(product, assign)

            # Coarsen node features
            x = torch.matmul(assign.transpose(0, 1), latent)
            x = F.normalize(x, dim = 1)

            # Coarsen the adjacency
            adj = torch.matmul(torch.matmul(assign.transpose(0, 1), adj), assign)
            adj = adj / torch.sum(adj)

            # New latent by graph convolution
            latent = torch.tanh(self.middle_encoder[level](torch.matmul(adj, x)))

            # Extended latent
            extended_latent = torch.matmul(product, latent)
            all_latents.append(extended_latent)

        # Normalization
        if self.use_norm == True:
            for idx in range(len(all_latents)):
                all_latents[idx] = all_latents[idx] / torch.norm(all_latents[idx], p = 2)

        # Concatenate all resolutions
        representation = torch.cat(all_latents, dim = 1)
        x = representation

        # Mixing multiple resolutions
        x = torch.relu(self.mix_1(x))
        x = torch.relu(self.mix_2(x))

        # +------------------------------------------------------------------------------------------+
        # | Attention model module, self-attention variant with query, key, and value the same input |
        # +------------------------------------------------------------------------------------------+
        x = x.view(-1, self.window, self.n_nodes, x.size(1)) 
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(self.window, -1, x.size(3))#self.batch_size*self.n_nodes

        x, _ = self.self_attention(x, x, x)
        x = torch.transpose(x, 0, 2)
        x = self.linear_reduction(x)
        x = x.squeeze()
        x = torch.transpose(x, 0, 1)

        skip = skip.reshape(skip.size(0),-1)
                
        x = torch.cat([x,skip], dim=1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x)).squeeze()
        x_2d = x.view(-1)
        x = torch.sum(x_2d)

        return x, x_2d




# +----------------------------------------------------------+
# | Attention Temporal Multiresolution Graph Neural Networks |
# +----------------------------------------------------------+

class ATMGNN(nn.Module):
    def __init__(self, nfeat, nhid, nout, n_nodes, window, dropout, nhead = 1, num_clusters = [10, 5], use_norm = False):
        super(ATMGNN, self).__init__()
        
        self.window = window
        self.n_nodes = n_nodes
        self.nhid = nhid
        self.nfeat = nfeat
        self.nhead = nhead
        self.use_norm = use_norm

        # +--------------------------------------+
        # | Multiresolution Graph Networks (MGN) |
        # +--------------------------------------+

        # Bottom encoder
        self.bottom_encoder = MPNN_Encoder(nfeat, nhid, nhid, dropout)

        # Number of clusters
        self.num_clusters = num_clusters

        # Multiresolution construction
        self.middle_linear = nn.ModuleList()
        self.middle_encoder = nn.ModuleList()

        for size in self.num_clusters:
            self.middle_linear.append(nn.Linear(nhid, size))
            self.middle_encoder.append(nn.Linear(nhid, nhid))

        # Mixing multiple resolutions together
        self.mix_1 = nn.Linear((len(self.num_clusters) + 1) * nhid, 512)
        self.mix_2 = nn.Linear(512, (len(self.num_clusters) + 1) * nhid)

        # +--------------------------+
        # | Attention/Self-Attention |
        # +--------------------------+

        self.self_attention = nn.MultiheadAttention((len(self.num_clusters) + 1) * nhid, self.nhead, dropout=dropout)
        self.linear_reduction = nn.Linear(self.window, 1)
        
        self.fc1 = nn.Linear((len(self.num_clusters) + 1) * nhid + window * nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nout)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        
    def forward(self, adj, x):
        lst = list()
        skip = x.view(-1, self.window, self.n_nodes, self.nfeat)
        skip = torch.transpose(skip, 1, 2).reshape(-1, self.window, self.nfeat)

        # +--------------------------------------+
        # | Multiresolution Graph Networks (MGN) |
        # +--------------------------------------+
        x = x.view(-1, self.nfeat)

        # All latents
        all_latents = []

        # Bottom encoder
        bottom_latent = self.bottom_encoder(adj, x)
        all_latents.append(bottom_latent)

        # Product of all assignment matrices
        product = None

        # Multiresolution construction
        adj = adj.to_dense()
        latent = bottom_latent

        for level in range(len(self.num_clusters)):
            size = self.num_clusters[level]

            # Assignment matrix
            assign = self.middle_linear[level](latent)
            assign = F.gumbel_softmax(assign, tau = 1, hard = True, dim = 1)

            # Update product
            if level == 0:
                product = assign
            else:
                product = torch.matmul(product, assign)

            # Coarsen node features
            x = torch.matmul(assign.transpose(0, 1), latent)
            x = F.normalize(x, dim = 1)

            # Coarsen the adjacency
            adj = torch.matmul(torch.matmul(assign.transpose(0, 1), adj), assign)
            adj = adj / torch.sum(adj)

            # New latent by graph convolution
            latent = torch.tanh(self.middle_encoder[level](torch.matmul(adj, x)))

            # Extended latent
            extended_latent = torch.matmul(product, latent)
            all_latents.append(extended_latent)

        # Normalization
        if self.use_norm == True:
            for idx in range(len(all_latents)):
                all_latents[idx] = all_latents[idx] / torch.norm(all_latents[idx], p = 2)

        # Concatenate all resolutions
        representation = torch.cat(all_latents, dim = 1)
        x = representation

        # Mixing multiple resolutions
        x = torch.relu(self.mix_1(x))
        x = torch.relu(self.mix_2(x))

        # +------------------------------------------------------------------------------------------+
        # | Attention model module, self-attention variant with query, key, and value the same input |
        # +------------------------------------------------------------------------------------------+
        x = x.view(-1, self.window, self.n_nodes, x.size(1)) 
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(self.window, -1, x.size(3))#self.batch_size*self.n_nodes

        x, _ = self.self_attention(x, x, x)
        x = torch.transpose(x, 0, 2)
        x = self.linear_reduction(x)
        x = x.squeeze()
        x = torch.transpose(x, 0, 1)

        skip = skip.reshape(skip.size(0),-1)
                
        x = torch.cat([x,skip], dim=1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x)).squeeze()
        x = x.view(-1)

        return x