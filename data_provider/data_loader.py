import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, flag, args):

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
        self.__read_data__()


    def __read_data__(self):
        df = pd.read_csv(self.path + "/data/" + self.city + "/df_grouped_1000m_" + self.sample_time + ".csv")

        self.min_tile_id = df['tile_ID_origin'].min()
        df['tile_ID_origin'] -= df['tile_ID_origin'].min()
        df['tile_ID_destination'] -= df['tile_ID_destination'].min()

        t = -1
        time = set()
        x_axis = int(df['tile_ID_origin'].max())+1
        y_axis = int(df['tile_ID_destination'].max())+1
        od_matrix = np.zeros([len(df['starttime'].unique()), x_axis, y_axis])

        for row in df.itertuples():
            if(row.starttime not in time):
                time.add(row.starttime)
                t += 1
            od_matrix[t, int(row.tile_ID_origin), int(row.tile_ID_destination)] = row.flow

        for i in range(od_matrix.shape[0]):
            np.fill_diagonal(od_matrix[i, :, :],0)

        od_sum = np.sum(od_matrix, axis=0)
        od_matrix = od_matrix[:, ~(od_sum==0).all(1), :]
        od_matrix = od_matrix[:, :, ~(od_sum.T==0).all(1)]

        self.tile_index = [i for i, x in enumerate(~(od_sum==0).all(1)) if x]
        self.empty_indices = [i for i, x in enumerate((od_sum==0).all(1)) if x]

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

    def get_tile_index(self):
        return self.tile_index, self.min_tile_id

    def get_empty(self):
        return self.empty_indices

    def get_key_indices(self):
        indices = []
        for i in range(self.num_tiles**2):
          index = []
          start = i // self.num_tiles
          end = i % self.num_tiles
          for j in range(self.num_tiles):
            index.append(start*self.num_tiles + j)
            index.append(end + self.num_tiles*j)
          index.remove(i)
          indices.append(sorted(index))

        return indices
