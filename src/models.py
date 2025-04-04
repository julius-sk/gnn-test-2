import os
os.environ["DGLBACKEND"] = "pytorch"
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv, GraphConv, GATConv, GATv2Conv
from dgl.dataloading import (
    as_edge_prediction_sampler,
    DataLoader,
    MultiLayerFullNeighborSampler,
    negative_sampler,
    NeighborSampler,
)


class GCN(nn.Module):
    def __init__(self, in_size, hid_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_size, hid_size, allow_zero_in_degree=True))
        self.layers.append(GraphConv(hid_size, hid_size, allow_zero_in_degree=True))
        self.layers.append(GraphConv(hid_size, hid_size, allow_zero_in_degree=True))
        
        self.hid_size = hid_size
        self.predictor = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )

    def forward(self, pair_graph, neg_pair_graph, blocks, x, edge_weight):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h, edge_weight=edge_weight[l] if edge_weight is not None else None)
            if l != len(self.layers) - 1:
                h = F.relu(h)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, device, batch_size):
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device
        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for _, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y





class GraphSAGE(nn.Module):
    def __init__(self, in_size, hid_size, weighted=False):
        super().__init__()
        self.layers = nn.ModuleList()
        if weighted:
            norm = lambda x: F.normalize(x, p=2.0, dim=-1)
            self.layers.append(SAGEConv(in_size, hid_size, "mean", norm=norm))
            self.layers.append(SAGEConv(hid_size, hid_size, "mean", norm=norm))
            self.layers.append(SAGEConv(hid_size, hid_size, "mean", norm=norm))
        else:
            self.layers.append(SAGEConv(in_size, hid_size, "mean"))
            self.layers.append(SAGEConv(hid_size, hid_size, "mean"))
            self.layers.append(SAGEConv(hid_size, hid_size, "mean"))
        self.hid_size = hid_size
        self.predictor = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )

    def forward(self, pair_graph, neg_pair_graph, blocks, x, edge_weight):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h, edge_weight=edge_weight[l] if edge_weight is not None else None)
            if l != len(self.layers) - 1:
                h = F.relu(h)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, device, batch_size):
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device
        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for _, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y



class GAT(nn.Module):
    def __init__(self, in_size, hid_size):
        super().__init__()
        self.layers = nn.ModuleList()

        self.out_channels = hid_size
        self.num_heads = 4

        self.layers.append(GATConv(in_size, hid_size, num_heads=self.num_heads, allow_zero_in_degree=True))
        self.layers.append(GATConv(hid_size * self.num_heads, hid_size, num_heads=self.num_heads, allow_zero_in_degree=True))
        self.layers.append(GATConv(hid_size * self.num_heads, hid_size, num_heads=self.num_heads, allow_zero_in_degree=True))
        
        self.projection = nn.Linear(hid_size * self.num_heads, hid_size)

        self.predictor = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )

    def forward(self, pair_graph, neg_pair_graph, blocks, x, edge_weight=None):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h).flatten(1)
            h = F.elu(h)
        
        h = self.projection(h)
        
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, device, batch_size):
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device
        
        for l, layer in enumerate(self.layers):
            out_feat_size = self.out_channels * self.num_heads
            
            y = torch.empty(
                g.num_nodes(),
                out_feat_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            
            feat = feat.to(device)
            for _, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x).flatten(1)
                h = F.elu(h)
                y[output_nodes] = h.to(buffer_device)
            
            feat = y 
        
        y = self.projection(y.to(device))
        
        return y


class GATv2(nn.Module):
    def __init__(self, in_size, hid_size):
        super().__init__()
        self.layers = nn.ModuleList()

        self.out_channels = hid_size
        self.num_heads = 4

        self.layers.append(GATv2Conv(in_size, hid_size, num_heads=self.num_heads, allow_zero_in_degree=True))
        self.layers.append(GATv2Conv(hid_size * self.num_heads, hid_size, num_heads=self.num_heads, allow_zero_in_degree=True))
        self.layers.append(GATv2Conv(hid_size * self.num_heads, hid_size, num_heads=self.num_heads, allow_zero_in_degree=True))
        
        self.projection = nn.Linear(hid_size * self.num_heads, hid_size)

        self.predictor = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )

    def forward(self, pair_graph, neg_pair_graph, blocks, x, edge_weight=None):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h).flatten(1)
            h = F.elu(h)
        
        h = self.projection(h)
        
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, device, batch_size):
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device
        
        for l, layer in enumerate(self.layers):
            out_feat_size = self.out_channels * self.num_heads
            
            y = torch.empty(
                g.num_nodes(),
                out_feat_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            
            feat = feat.to(device)
            for _, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x).flatten(1)
                h = F.elu(h)
                y[output_nodes] = h.to(buffer_device)
            
            feat = y

        y = self.projection(y.to(device))
        
        return y







