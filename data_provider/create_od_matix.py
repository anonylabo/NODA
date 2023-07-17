import pandas as pd
import numpy as np
from utils.dataset_utils import get_normalized_adj


def create_od_matrix(args):
    df = pd.read_csv(args.path + "/data/" + args.city + "/df_grouped_1000m_" + args.sample_time + ".csv")

    min_tile_id = df['tile_ID_origin'].min()
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

    if args.model=='GTFormer':
        key_indices = []
        for i in range(args.num_tiles**2):
            index = []
            start = i // args.num_tiles
            end = i % args.num_tiles
            for j in range(args.num_tiles):
                index.append(start*args.num_tiles + j)
                index.append(end + args.num_tiles*j)
            index.remove(i)
            key_indices.append(sorted(index))

    else:
        A = od_matrix.sum(axis=0)
        A_hat = get_normalized_adj(A)

    tile_index = [i for i, x in enumerate(~(od_sum==0).all(1)) if x]
    empty_indices = [i for i, x in enumerate((od_sum==0).all(1)) if x]

    
    if args.model=='GTFormer':
        return od_matrix, min_tile_id, tile_index, empty_indices, key_indices
    else:
        return od_matrix, min_tile_id, tile_index, empty_indices, A_hat