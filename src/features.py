import sys
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import haversine_distance, euclidean_distance, manhattan_distance

from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from src.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()



TARGET_COLUMN = 'trip_duration'
PLOT_PATH = Path("reports/figures/target_distribution.png")


## Functions applied on target column
def convert_target_to_minutes(dataframe: pd.DataFrame, target_column: str) -> pd.DataFrame:
    # convert the target into minutes
    dataframe.loc[:,target_column] = dataframe[target_column] / 60
    logger.info('Target column converted from seconds into minutes')
    return dataframe

def drop_above_two_hundred_minutes(dataframe: pd.DataFrame, target_column: str) -> pd.DataFrame:
    # filter rows where target is less than 200
    filter_series = dataframe[target_column] <= 200
    new_dataframe = dataframe.loc[filter_series,:].copy()
    # max value of target column to checjk the outliers are removed
    max_value = new_dataframe[target_column].max()
    logger.info(f'The max value in target column after transformation is {max_value} and the state of transformation is {max_value <= 200}')
    if max_value <= 200:
        return new_dataframe
    else:
        raise ValueError('Outlier target values not removed from the data')        


def plot_target(dataframe: pd.DataFrame, target_column: str, save_path: str):
    # plot the density plot of the target after transformation
    sns.kdeplot(data=dataframe, x=target_column)
    plt.title(f'Distribution of {target_column}')
    # save the plot at the destination path
    plt.savefig(save_path)
    logger.info('Distribution plot saved at destination')
    
    
def drop_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    logger.info(f'Columns in data before removal are {list(dataframe.columns)}')
    # drop columns from train and val data
    if 'dropoff_datetime' in dataframe.columns:
        columns_to_drop = ['id','dropoff_datetime','store_and_fwd_flag']
        # dropping the columns from dataframe
        dataframe_after_removal = dataframe.drop(columns=columns_to_drop)
        list_of_columns_after_removal = list(dataframe_after_removal.columns)
        logger.info(f'Columns in data after removal are {list_of_columns_after_removal}')
        # verifying if columns dropped
        logger.info(f"Columns {', '.join(columns_to_drop)} dropped from data  verify={columns_to_drop not in list_of_columns_after_removal}")
        return dataframe_after_removal
    # drop columns from the test data
    else:
        columns_to_drop = ['id','store_and_fwd_flag']
        # dropping the columns from dataframe
        dataframe_after_removal = dataframe.drop(columns=columns_to_drop)
        list_of_columns_after_removal = list(dataframe_after_removal.columns)
        # verifying if columns dropped
        logger.info(f"Columns {', '.join(columns_to_drop)} dropped from data  verify={columns_to_drop not in list_of_columns_after_removal}")
        return dataframe_after_removal


def make_datetime_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    # copy the original dataframe
    new_dataframe = dataframe.copy()
    # number of rows and column before transformation
    original_number_of_rows, original_number_of_columns = new_dataframe.shape
    
    # convert the column to datetime column
    new_dataframe['pickup_datetime'] = pd.to_datetime(new_dataframe['pickup_datetime'])
    logger.info(f'pickup_datetime column converted to datetime {new_dataframe["pickup_datetime"].dtype}')
    
    # do the modifications
    new_dataframe.loc[:,'pickup_hour'] = new_dataframe['pickup_datetime'].dt.hour 
    new_dataframe.loc[:,'pickup_date'] = new_dataframe['pickup_datetime'].dt.day
    new_dataframe.loc[:,'pickup_month'] = new_dataframe['pickup_datetime'].dt.month
    new_dataframe.loc[:,'pickup_day'] = new_dataframe['pickup_datetime'].dt.weekday
    new_dataframe.loc[:,'is_weekend'] = new_dataframe.apply(lambda row: row['pickup_day'] >= 5,axis=1).astype('int')
    
    # drop the redundant date time column
    new_dataframe = new_dataframe.drop(columns=['pickup_datetime'])
    logger.info(f'pickup_datetime column dropped  verify={"pickup_datetime" not in new_dataframe.columns}')
    
    # number of rows and columns after transformation
    transformed_number_of_rows, transformed_number_of_columns = new_dataframe.shape
    logger.info(f'The number of columns increased by 4 {transformed_number_of_columns == (original_number_of_columns + 5 - 1)}')
    logger.info(f'The number of rows remained the same {original_number_of_rows == transformed_number_of_rows}')
    return new_dataframe


def remove_passengers(dataframe: pd.DataFrame) -> pd.DataFrame:
    # make the list of passenger to keep
    passengers_to_include = list(range(1,7))
    # filter out rows which matches exavctly the passengers in the list
    new_dataframe_filter = dataframe['passenger_count'].isin(passengers_to_include)
    # filter the dataframe
    new_dataframe = dataframe.loc[new_dataframe_filter,:]
    # list of unique passenger values in the passenger_count column
    unique_passenger_values = list(np.sort(new_dataframe['passenger_count'].unique()))
    logger.info(f'The unique passenger list is {unique_passenger_values}  verify={passengers_to_include == unique_passenger_values}')
    return new_dataframe


def input_modifications(dataframe: pd.DataFrame) -> pd.DataFrame:
    # drop the columns in input data
    new_df = drop_columns(dataframe)
    # remove the rows having excluded passenger values
    df_passengers_modifications = remove_passengers(new_df)
    # add datetime features to data
    df_with_datetime_features = make_datetime_features(df_passengers_modifications)
    logger.info('Modifications with input features complete')
    return df_with_datetime_features

   
def target_modifications(dataframe: pd.DataFrame, target_column: str=TARGET_COLUMN) -> pd.DataFrame:
    # convert the target column from seconds to minutes
    minutes_dataframe = convert_target_to_minutes(dataframe,target_column)
    # remove target values greater than 200
    target_outliers_removed_df = drop_above_two_hundred_minutes(minutes_dataframe,target_column)
    # plot the target column
    plot_target(dataframe=target_outliers_removed_df,target_column=target_column,
                save_path=PLOT_PATH)
    logger.info('Modifications with the target feature complete')
    return target_outliers_removed_df

# read the dataframe from location
def read_data(data_path):
    df = pd.read_csv(data_path)
    return df

# save the dataframe to location
def save_data(dataframe: pd.DataFrame,save_path: Path):
    dataframe.to_csv(save_path,index=False)
    
    
# TODO 1. Make a function to read the dataframe from the dvc.yaml file
# TODO 2. Add Logging Functionality to each function
# TODO 3. Run the code in notebook mode to test with print statements
# ? Should logging be added to each function or the main function for specific steps


def modify_features(data_path,filename):
    # read the data into dataframe
    df = read_data(data_path)
    # do the modifications on the input data
    df_input_modifications = input_modifications(dataframe=df)
    # check whether the input file has target column
    if (filename == "train.csv") or (filename == "val.csv"):
        df_final = target_modifications(dataframe=df_input_modifications)  
    else:
        df_final = df_input_modifications
        
    return df_final
        



new_feature_names = ['haversine_distance',
                     'euclidean_distance',
                     'manhattan_distance']

build_features_list = [haversine_distance,
                       euclidean_distance,
                       manhattan_distance]


def implement_distances(dataframe:pd.DataFrame, 
                        lat1:pd.Series, 
                        lon1:pd.Series, 
                        lat2:pd.Series, 
                        lon2:pd.Series) -> pd.DataFrame:
    dataframe = dataframe.copy()
    for ind in range(len(build_features_list)):
        func = build_features_list[ind]
        dataframe[new_feature_names[ind]] = func(lat1,lon1,
                                                 lat2,lon2)
    
    return dataframe
        

@app.command()
def modify(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path,
    output_path: Path = PROCESSED_DATA_DIR / "transformations"
):
    df_final = modify_features(input_path,input_path.name)
    output_path.mkdir(parents=True,exist_ok=True)
    # save the data
    save_data(df_final,output_path / input_path.name)
    logger.info(f'{output_path / input_path.name} saved at the destination folder')






@app.command()
def build(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "transformations",
    output_path: Path = PROCESSED_DATA_DIR / "build-features",
    # -----------------------------------------
):
    output_path.mkdir(parents=True, exist_ok=True)
    for inp in input_path.iterdir():
        if not inp.is_file():
            continue
        df = read_data(inp)
        df = implement_distances(
            dataframe=df,
            lat1=df['pickup_latitude'],
            lon1=df['pickup_longitude'],
            lat2=df['dropoff_latitude'],
            lon2=df['dropoff_longitude'],
        )
        save_data(df, output_path / inp.name)
        logger.info(f'{output_path / inp.name} saved at the destination folder')

 
    
if __name__ == "__main__":
    app()
