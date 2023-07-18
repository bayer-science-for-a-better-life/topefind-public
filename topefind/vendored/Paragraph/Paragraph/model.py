import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from topefind.vendored.Paragraph.Paragraph.EGNN import EGNN


class EGNN_Model(nn.Module):
    '''
    Paragraph uses equivariant graph layers with skip connections
    '''
    def __init__(
        self,
        num_feats,
        edge_dim = 1,
        output_dim = 1,
        graph_hidden_layer_output_dims = None,
        linear_hidden_layer_output_dims = None,
        update_coors = False,
        dropout = 0.0,
        m_dim = 16
    ):
        super(EGNN_Model, self).__init__()

        self.input_dim = num_feats
        self.output_dim = output_dim
        current_dim = num_feats

        # these will store the different layers of out model
        self.graph_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()

        # model with 1 standard EGNN and single dense layer if no architecture provided
        if graph_hidden_layer_output_dims == None: graph_hidden_layer_output_dims = [num_feats]
        if linear_hidden_layer_output_dims == None: linear_hidden_layer_output_dims = []

        # graph layers
        for hdim in graph_hidden_layer_output_dims:
            self.graph_layers.append(EGNN(dim = current_dim,
                                          edge_dim = edge_dim,
                                          update_coors = update_coors,
                                          dropout = dropout,
                                          m_dim = m_dim))
            current_dim = hdim

        # dense layers
        for hdim in linear_hidden_layer_output_dims:
            self.linear_layers.append(nn.Linear(in_features = current_dim,
                                                out_features = hdim))
            current_dim = hdim

        # final layer to get to per-node output
        self.linear_layers.append(nn.Linear(in_features = current_dim, out_features = output_dim))


    def forward(self, feats, coors, edges, mask=None):

        # graph layers
        for layer in self.graph_layers:
            feats = F.hardtanh(layer(feats, coors, edges, mask))

        # dense layers
        for layer in self.linear_layers[:-1]:
            feats = F.hardtanh(layer(feats))

        # output (i.e. prediction)
        feats = self.linear_layers[-1](feats)

        return feats
