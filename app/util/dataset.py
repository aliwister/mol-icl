from torch.utils.data import Dataset

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
    