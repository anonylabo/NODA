import pandas as pd
import numpy as np
from utils.dataset_utils import get_normalized_adj


def create_od_matrix(args):
    df = pd.read_csv(args.path + "/data/" + args.city + "/df_grouped_1000m_" + args.sample_time + ".csv")

    minites = {'60min':60, '45min':45, '30min':30, '15min':15}
    minite = minites[args.sample_time]

    #Calculate how many time intervals each time is counted from the beginning
    df['start'] = [df['starttime'][0] for _ in range(len(df))]
    df['start'] = pd.to_datetime(df['start'])
    df['starttime'] = pd.to_datetime(df['starttime'])
    df['dif'] = df['starttime'] - df['start']
    df['dif'] = df['dif'].dt.total_seconds().round()/(minite*60)
    df['dif'] = df['dif'].astype('int')

    min_tile_id = df['tile_ID_origin'].min()
    df['tile_ID_origin'] -= df['tile_ID_origin'].min()
    df['tile_ID_destination'] -= df['tile_ID_destination'].min()

    #Create an empty ODmatrix
    x_axis = int(df['tile_ID_origin'].max())+1
    y_axis = int(df['tile_ID_destination'].max())+1
    od_matrix = np.zeros([df['dif'].max()+1, x_axis, y_axis])

    #Substitute each flow
    for row in df.itertuples():
        od_matrix[row.dif, int(row.tile_ID_origin), int(row.tile_ID_destination)] = row.flow

    #The diagonal component is not regarded as flow, so it is set to 0
    for i in range(od_matrix.shape[0]):
        np.fill_diagonal(od_matrix[i, :, :],0)

    #Remove origin-destination pairs whose flow is 0 at all times to make the calculation lighter
    od_sum = np.sum(od_matrix, axis=0)
    od_matrix = od_matrix[:, ~(od_sum==0).all(1), :]
    od_matrix = od_matrix[:, :, ~(od_sum==0).all(1)]

    #Get indices of M in KVR for GTFformer
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

    #Get adjacency matrix for CrowdNet
    else:
        A = od_matrix.sum(axis=0)
        A_hat = get_normalized_adj(A)

    #For restore ODmatrix
    empty_indices = [i for i, x in enumerate((od_sum==0).all(1)) if x]

    
    if args.model=='GTFormer':
        return od_matrix, min_tile_id, empty_indices, key_indices
    else:
        return od_matrix, min_tile_id, empty_indices, A_hat