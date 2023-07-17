import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, flag, args, od_matrix):

        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.path = args.path
        self.tile_size = args.tile_size
        self.sample_time = args.sample_time
        self.city = args.city
        day_steps = {'60min':24, '45min':32, '30min':48, '15min':96}
        self.day_step = day_steps[args.sample_time]
        self.num_tiles = args.num_tiles
        self.__read_data__(od_matrix)


    def __read_data__(self, od_matrix):

        border1s = [0, self.day_step*138, self.day_step*173]
        border2s = [self.day_step*138, self.day_step*173, self.day_step*183]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        od_matrix_ = od_matrix[border1:border2]

        self.data = np.zeros(((self.day_step - self.seq_len)*od_matrix_.shape[0]//self.day_step, self.seq_len+self.pred_len, od_matrix_.shape[-2], od_matrix_.shape[-1]))
        for i in range(od_matrix_.shape[0]//self.day_step):
          for j in range(self.day_step - self.seq_len):
            sta = i * self.day_step + j
            end = sta + self.seq_len + self.pred_len
            self.data[i * (self.day_step - self.seq_len) + j, :, :, :] = od_matrix_[sta:end]


    def __getitem__(self, index):

        seq_x = self.data[index, :-self.pred_len]
        seq_y = self.data[index, -self.pred_len:]

        return seq_x, seq_y


    def __len__(self):
        return len(self.data)
