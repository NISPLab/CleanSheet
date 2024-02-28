from torch.utils.data import Dataset
import torch


class PoisonDataset(Dataset):
    def __init__(self, dataset, indices, target):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        self.target = target

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        x, y = self.dataset[self.indices[item]]
        # print(type(y))
        # print(y)
        # print(x.shape)
        # y = torch.tensor(self.target)
        # print(y.shape)
        # print(type(self.target))
        return x, self.target
