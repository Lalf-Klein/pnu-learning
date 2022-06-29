import torch


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, train_x, train_y):
        super().__init__()
        self.train_x = train_x
        self.train_y = train_y
    
    def __len__(self):
        return self.train_x.shape[0]
    
    def __getitem__(self, idx):
        return self.train_x.transpose(0, 3, 1, 2)[idx], self.train_y[idx]
    