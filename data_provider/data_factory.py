from data_provider.data_loader import MyDataset
from torch.utils.data import DataLoader

def data_provider(flag, args):

    shuffle_flag = False
    drop_last = False
    batch_size = args.batch_size

    data_set = MyDataset(flag, args)

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last,
        num_workers=2)

    tile_index, min_tile_id, empty_indices, param = data_set.get_some()

    return data_loader, empty_indices, min_tile_id, tile_index, param