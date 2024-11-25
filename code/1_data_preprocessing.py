
# Import all required modules
import pandas as pd
import geopandas as gpd
import numpy as np
import h3
from data_processing_helpers import * 


# read input data
print("Reading input data...")
df = pd.read_csv("../input/dataset_240526_res_1.csv") # primary data
all_ports = pd.read_csv("../input/all_ports.csv")    # port particulars data
all_ports.loc[all_ports['country'] == 'Namibia', 'country_code'] = 'NA' # Nambia with port code NA is eroneously read as a Nan
vessel_particulars = pd.read_csv("../input/vessel_particulars.csv") # vessel particulars data	

# extract required columns and merge with teu data
print("Data preprocessing...")
trips_df = df[["imo","timestamp","latitude","longitude","locode_last","locode_next"]]
trips_df= trips_df.rename(columns={"latitude":"pos_latitude","longitude":"pos_longitude"})
trips_df = pd.merge( left= trips_df, right= vessel_particulars[["imo","teu"]], on= 'imo',  how= 'left')

# Create vessel classification based o teu
bins = [0, 1000, 2000, 3000, 5100, 10000, 15500, float('inf')]
labels = ['Small', 'Feeder', 'Feedermax', 'Panamax', 'Post-Panamax', 'New-Panamax', 'ULCV']
trips_df['classification'] = pd.cut(trips_df['teu'], bins=bins, labels=labels, right=False)

# remove vessels with unknown classification
trips_df = trips_df[trips_df["classification"].notna()]
trips_df = trips_df.sort_values(by= ["imo","timestamp"])

# get h3 cell for each position(latitude, longitude) using the function get_h3_cell
print("Adding h3 cells...")
trips_df['cell_h3'] = trips_df.apply(get_h3_cell, resolution=1, axis=1)
trips_df=trips_df.reset_index(drop=True)

# remove consecutive positional duplicate records from each imo group
print("Removing consecutive positional duplicate records...")
cols = trips_df.columns.tolist()  
trips_df = trips_df.groupby('imo')[cols].apply(remove_consecutive_duplicates,col_1='cell_h3', col_2='locode_last' ).reset_index(drop=True) 
trips_df.to_csv("../temp/trips_df_without_duplicates.csv", index=False)
print("First intermediate result saved in temp folder as trips_df_without_duplicates.csv")

# clean anomalous positional records using the function clean_anomalies
print("Cleaning anomalous positional records...")
cols= trips_df.columns.tolist()
trips_df = trips_df.groupby('imo')[cols].apply(clean_anomalies, ring_size=3).reset_index(drop=True)
trips_df['imo']=trips_df['imo'].astype(int)

# extract month and year from timestamp
print("Additional preprocessing...")
trips_df['timestamp'] = pd.to_datetime(trips_df['timestamp'])
trips_df['month_str'] = trips_df['timestamp'].dt.strftime('%b')


# get country code from port locode: first 2 characters
trips_df['country_code_last'] = trips_df['locode_last'].str[:2]
trips_df['country_code_next'] = trips_df['locode_next'].str[:2]

# select only relevant columns and save second intermediate result
trips_df= trips_df[["imo","timestamp","classification","cell_h3","month_str", "pos_latitude", "pos_longitude", "locode_last","locode_next", "country_code_last", "country_code_next"]]
trips_df.to_csv("../temp/trips_df_cleaned.csv", index=False)
print("Cleaned data saved in temp folder as trips_df_cleaned.csv")

# summary of trips using the function summarize_trips and save final result
print("Summarizing trips...")
trips_summary_df = summarize_trips(trips_df) 
trips_summary_df.to_csv("../temp/trips_summary_df.csv", index=False)
print("Final result saved in temp folder as trips_summary_df.csv")
print("Data preprocessing completed successfully!")

## END
