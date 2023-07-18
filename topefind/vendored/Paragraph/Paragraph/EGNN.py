# original code: https://github.com/lucidrains/egnn-pytorch

import torch
from torch import nn, einsum, broadcast_tensors
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helper function
def exists(val):
    return val is not None


# swish activation fallback
class Swish_(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

SiLU = nn.SiLU if hasattr(nn, 'SiLU') else Swish_

# standard EGNN layer
class EGNN(nn.Module):
    def __init__(
        self,
        dim,
        edge_dim = 0,
        m_dim = 16,
        fourier_features = 0,
        num_nearest_neighbors = 0,
        dropout = 0.0,
        init_eps = 1e-3,
        norm_feats = False,
        norm_coors = False,
        update_feats = True,
        update_coors = False,
        only_sparse_neighbors = False,
        valid_radius = float('inf'),
        m_pool_method = 'sum'
    ):
        super().__init__()
        assert m_pool_method in {'sum', 'mean'}, 'pool method must be either sum or mean'
        assert update_feats or update_coors, 'you must update either features, coordinates, or both'

        self.fourier_features = fourier_features

        edge_input_dim = (fourier_features * 2) + (dim * 2) + edge_dim + 1
        dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim * 2),
            dropout,
            SiLU(),
            nn.Linear(edge_input_dim * 2, m_dim),
            SiLU()
        )

        self.node_norm = nn.LayerNorm(dim) if norm_feats else nn.Identity()
        self.coors_norm = CoorsNorm() if norm_coors else nn.Identity()

        self.m_pool_method = m_pool_method

        self.node_mlp = nn.Sequential(
            nn.Linear(dim + m_dim, dim * 2),
            dropout,
            SiLU(),
            nn.Linear(dim * 2, dim),
        ) if update_feats else None

        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            dropout,
            SiLU(),
            nn.Linear(m_dim * 4, 1)
        ) if update_coors else None

        self.num_nearest_neighbors = num_nearest_neighbors
        self.only_sparse_neighbors = only_sparse_neighbors
        self.valid_radius = valid_radius

        self.init_eps = init_eps
        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            # seems to be needed to keep the network from exploding to NaN with greater depths
            nn.init.normal_(module.weight, std = self.init_eps)

    def forward(self, feats, coors, edges = None, mask = None, adj_mat = None):
        b, n, d, device, fourier_features, num_nearest, valid_radius, only_sparse_neighbors = *feats.shape, feats.device, self.fourier_features, self.num_nearest_neighbors, self.valid_radius, self.only_sparse_neighbors

        if exists(mask):
            num_nodes = mask.sum(dim = -1)

        use_nearest = num_nearest > 0 or only_sparse_neighbors

        rel_coors = rearrange(coors, 'b i d -> b i () d') - rearrange(coors, 'b j d -> b () j d')
        rel_dist = (rel_coors ** 2).sum(dim = -1, keepdim = True)

        i = j = n

        if use_nearest:
            ranking = rel_dist[..., 0]

            if exists(mask):
                rank_mask = mask[:, None, :] * mask[:, None, :]
                ranking.masked_fill_(~rank_mask, 1e5)

            if exists(adj_mat):
                if len(adj_mat.shape) == 2:
                    adj_mat = repeat(adj_mat.clone(), 'i j -> b i j', b = b)

                if only_sparse_neighbors:
                    num_nearest = int(adj_mat.float().sum(dim = -1).max().item())
                    valid_radius = 0

                self_mask = rearrange(torch.eye(n, device = device, dtype = torch.bool), 'i j -> () i j')

                adj_mat = adj_mat.masked_fill(self_mask, False)
                ranking.masked_fill_(self_mask, -1.)
                ranking.masked_fill_(adj_mat, 0.)

            nbhd_ranking, nbhd_indices = ranking.topk(num_nearest, dim = -1, largest = False)

            nbhd_mask = nbhd_ranking <= valid_radius

            rel_coors = batched_index_select(rel_coors, nbhd_indices, dim = 2)
            rel_dist = batched_index_select(rel_dist, nbhd_indices, dim = 2)

            if exists(edges):
                edges = batched_index_select(edges, nbhd_indices, dim = 2)

            j = num_nearest

        if fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist, num_encodings = fourier_features)
            rel_dist = rearrange(rel_dist, 'b i j () d -> b i j d')

        if use_nearest:
            feats_j = batched_index_select(feats, nbhd_indices, dim = 1)
        else:
            feats_j = rearrange(feats, 'b j d -> b () j d')

        feats_i = rearrange(feats, 'b i d -> b i () d')
        feats_i, feats_j = broadcast_tensors(feats_i, feats_j)

        edge_input = torch.cat((feats_i, feats_j, rel_dist), dim = -1)

        if exists(edges):
            edge_input = torch.cat((edge_input, edges), dim = -1)

        m_ij = self.edge_mlp(edge_input)

        if exists(mask):
            mask_i = rearrange(mask, 'b i -> b i ()')

            if use_nearest:
                mask_j = batched_index_select(mask, nbhd_indices, dim = 1)
                mask = (mask_i * mask_j) & nbhd_mask
            else:
                mask_j = rearrange(mask, 'b j -> b () j')
                mask = mask_i * mask_j

        if exists(self.coors_mlp):
            coor_weights = self.coors_mlp(m_ij)
            coor_weights = rearrange(coor_weights, 'b i j () -> b i j')

            rel_coors = self.coors_norm(rel_coors)

            if exists(mask):
                coor_weights.masked_fill_(~mask, 0.)

            coors_out = einsum('b i j, b i j c -> b i c', coor_weights, rel_coors) + coors
        else:
            coors_out = coors

        if exists(self.node_mlp):
            if exists(mask):
                m_ij_mask = rearrange(mask, '... -> ... ()')
                m_ij = m_ij.masked_fill(~m_ij_mask, 0.)

            if self.m_pool_method == 'mean':
                if exists(mask):
                    # masked mean
                    mask_sum = m_ij_mask.sum(dim = -2)
                    m_i = safe_div(m_ij.sum(dim = -2), mask_sum)
                else:
                    m_i = m_ij.mean(dim = -2)

            elif self.m_pool_method == 'sum':
                m_i = m_ij.sum(dim = -2)

            normed_feats = self.node_norm(feats)
            node_mlp_input = torch.cat((normed_feats, m_i), dim = -1)
            node_out = self.node_mlp(node_mlp_input) + feats
        else:
            node_out = feats

        return node_out
    