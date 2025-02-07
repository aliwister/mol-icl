from torch.utils.data import Dataset
import torch
from torch_geometric.data import InMemoryDataset

class PubChemDataset(InMemoryDataset):
    def __init__(self, path):
        super(PubChemDataset, self).__init__()
        self.data, self.slices = torch.load(path)
    
    def __getitem__(self, idx):
        return self.get(idx)


class GraphTextDataset(Dataset):
    def __init__(self, graphs, texts):
        assert len(graphs) == len(texts), "Graphs and texts must have the same length"
        self.graphs = graphs
        self.texts = texts

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        graph = self.graphs[index]
        text = self.texts[index]
        return graph, text

class GraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        graph = self.graphs[index]
        return graph

class GraphTextMorganDataset(Dataset):
    def __init__(self, graphs, texts, morgans, query_vecs=None):
        assert len(graphs) == len(texts), "Graphs and texts must have the same length"
        self.graphs = graphs
        self.texts = texts
        self.morgans = morgans
        self.query_vecs = query_vecs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        graph = self.graphs[index]
        text = self.texts[index]
        morgan = self.morgans[index]
        if (self.query_vecs is not None):
            query_vec = self.query_vecs[index]
            return index, graph, text, morgan, torch.tensor(query_vec, dtype=torch.float32)
        else:
            return index, graph, text, morgan    
    