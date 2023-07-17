from data_provider.data_loader import MyDataset
from torch.utils.data import DataLoader

def data_provider(flag, args, od_matrix):

    shuffle_flag = False
    drop_last = False
    batch_size = args.batch_size

    data_set = MyDataset(flag, args, od_matrix)

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last,
        num_workers=2)


    return data_loader