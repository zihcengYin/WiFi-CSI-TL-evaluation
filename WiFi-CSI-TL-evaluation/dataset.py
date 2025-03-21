import numpy as np
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

class CSI_Dataset(Dataset):
    def __init__(self, root_dir, modal='CSIamp'):
        self.root_dir = root_dir
        self.modal = modal
        self.data_list = glob.glob(root_dir + '\\*\\*.mat')
        self.folder = glob.glob(root_dir + '\\*\\')
        self.category = {self.folder[i].split('\\')[-2]: i for i in range(len(self.folder))}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('\\')[-2]]
        x = sio.loadmat(sample_dir)[self.modal]

        # 归一化
        x = (x - 42.3199) / 4.9802
        x = x[:, ::4]
        x = x.reshape(3, 114, 500)

        x = torch.FloatTensor(x)
        return x, y
