import os
import geopandas as gpd
import pandas as pd
import numpy as np
import ast
import requests
import shapely.wkt
from operator import itemgetter
import json
from urllib.request import urlopen



def load_dataset(city, tile_size, sample_time, dataset_directory):
    print("Loading data...")

    if city=='NYC':
        for month in range(4,10):
            if not os.path.isfile(dataset_directory + "20140" + str(month) + "-citibike-tripdata.zip"):
                url = "https://s3.amazonaws.com/tripdata/20140" + str(month)+ "-citibike-tripdata.zip"
                r = requests.get(url, allow_redirects=True)
                open(dataset_directory + "20140" + str(month) + "-citibike-tripdata.zip", 'wb').write(r.content)
                print("Downloaded month: ", month)

        print("data preprocessing...")
        zip_files = [f for f in os.listdir(dataset_directory) if f.endswith('.zip')]
        data = [pd.read_csv(dataset_directory+file_name) for file_name in zip_files]
        df = pd.concat(data)
    
    else:
        for year in ['2018', '2019']:
            for month in ['01','02','03','04','05','06','07','08','09','10','11','12']:
                if not os.path.isfile(dataset_directory + year + month + "-captialbikeshare-tripdata.zip"):
                    url = "https://s3.amazonaws.com/capitalbikeshare-data/" + year + month + "-capitalbikeshare-tripdata.zip"
                    r = requests.get(url, allow_redirects=True)
                    open(dataset_directory + year + month + "-capitalbikeshare-tripdata.zip", 'wb').write(r.content)
                    print("Downloaded month: ", month)
        
        print("data preprocessing...")
        zip_files = [f for f in os.listdir(dataset_directory) if f.endswith('.zip')]
        data = [pd.read_csv(dataset_directory+file_name) for file_name in zip_files]
        df = pd.concat(data)

        url = "https://gbfs.capitalbikeshare.com/gbfs/fr/station_information.json"
        response = urlopen(url)
        station_information = json.load(response)

        lat = []
        lon = []
        short_name = []
        for i in range(len(station_information['data']['stations'])):
            lat.append(station_information['data']['stations'][i]['lat'])
            lon.append(station_information['data']['stations'][i]['lon'])
            short_name.append(int(station_information['data']['stations'][i]['short_name']))

        station_df = pd.DataFrame({'lat':lat, 'lon':lon, 'short_name':short_name})
        station_df_sta = station_df.rename({'lat':'start station latitude', 'lon':'start station longitude', 'short_name':'Start station number'}, axis=1)
        station_df_end = station_df.rename({'lat':'end station latitude', 'lon':'end station longitude', 'short_name':'End station number'}, axis=1)

        df = pd.merge(df, station_df_sta, on='Start station number', how='left')
        df = pd.merge(df, station_df_end, on='End station number', how='left')

        df = df.drop(['Duration', 'Start station number', 'Start station', 'End station number', 'End station', 'Bike number', 'Member type'], axis=1)
        df = df.rename({'Start date':'starttime', 'End date':'stoptime'}, axis=1)
        df = df.dropna()


    tessellation = pd.read_csv(dataset_directory + city + "/Tessellation_" + str(tile_size) + "m.csv")
    tessellation['geometry'] = [shapely.wkt.loads(el) for el in tessellation.geometry]
    tessellation = gpd.GeoDataFrame(tessellation, geometry='geometry')

    list_positions = np.array([ast.literal_eval(el) for el in tessellation['position']])

    max_x = list_positions[:, 0].max()
    max_y = list_positions[:, 1].max()

    for i, y in enumerate(list_positions[:, 1]):
        list_positions[i, 1] = max_y - y

    tessellation['positions'] = list(sorted(list_positions, key=itemgetter(0)))


    df = df.drop(['tripduration', 'start station id',  'start station name', 'end station id', 'end station name', 'bikeid', 'usertype', 'birth year', 'gender'], axis=1)

    gdf_in = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['start station longitude'], df['start station latitude']),  crs='epsg:4326')
    gdf_in_join = gpd.sjoin(gdf_in, tessellation)
    gdf_in_join = gdf_in_join[['starttime',	'end station latitude', 'end station longitude', 'stoptime', 'tile_ID']]

    gdf_final = gpd.GeoDataFrame(gdf_in_join, geometry=gpd.points_from_xy(gdf_in_join['end station longitude'], gdf_in_join['end station latitude']),  crs='epsg:4326')
    gdf_final_join = gpd.sjoin(gdf_final, tessellation)
    gdf_final_join = gdf_final_join[['starttime', 'stoptime', 'tile_ID_left', 'tile_ID_right']]

    gdf_final_join = gdf_final_join.rename(columns={"tile_ID_left": "tile_ID_origin", "tile_ID_right": "tile_ID_destination"})
    gdf_final_join['starttime'] = pd.to_datetime(gdf_final_join['starttime'])
    gdf_final_join = gdf_final_join.sort_values(by='starttime')


    gdf_final_join['flow'] = 1
    gdf = gdf_final_join[['starttime', 'tile_ID_origin', 'tile_ID_destination', 'flow']]

    gdf_grouped = gdf.groupby([pd.Grouper(key='starttime', freq=sample_time), 'tile_ID_origin','tile_ID_destination']).sum()

    # Saving geodataframe
    gdf_grouped.to_csv(dataset_directory + city + "/df_grouped_" + str(tile_size) + "m_" + sample_time + ".csv")
