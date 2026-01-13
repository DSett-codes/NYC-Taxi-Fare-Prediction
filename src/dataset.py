from pathlib import Path
from zipfile import ZipFile
from loguru import logger
from tqdm import tqdm
import typer
import sys
from yaml import safe_load
from sklearn.model_selection import train_test_split
import threading
import pandas as pd
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR,INTERIM_DATA_DIR

app = typer.Typer()

def extract_zip_files(input_path: Path,output_path: Path):
    try:
        with ZipFile(file=input_path) as f:
            f.extractall(path=output_path)
            logger.info(f"{input_path} extracted successfully at the target path")
    except FileNotFoundError:
        logger.error(f"File not found in {input_path}")

def extract_raw_data(input_path: Path) -> pd.DataFrame :
    df = pd.read_csv(input_path)
    rows, columns = df.shape # type: ignore
    logger.info(f"{input_path.stem} data read having {rows} rows and {columns} columns.")
    return df

def train_val_split(data: pd.DataFrame,
                    test_size: float,
                    random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    train_data, val_data = train_test_split(data,
                                            test_size= test_size,
                                            random_state= random_state)
    logger.info(f'Data is split into train split with shape {train_data.shape} and val split with shape {val_data.shape}')
    logger.info(f'The parameter values are {test_size} for test_size and {random_state} for random_state')
    return train_data, val_data

def save_data(data: pd.DataFrame,output_path: Path):
    data.to_csv(output_path,index=False)
    logger.info(f'{output_path.stem + output_path.suffix} data saved successfully to the output folder')

def read_params(input_file):
    try:
        with open(input_file) as f:
            params_file = safe_load(f)
            
    except FileNotFoundError as e:
        logger.error('Parameters file not found, Switching to default values for train test split')
        default_dict = {'test_size': 0.25,
                        'random_state': None}
        # read the default_dictionary
        test_size = default_dict['test_size']
        random_state = default_dict['random_state']
        return test_size, random_state
        
    else:
        logger.info('Parameters file read successfully')
        # read the parameters from the parameters file
        test_size = params_file['make_dataset']['test_size']
        random_state = params_file['make_dataset']['random_state']
        return test_size, random_state

@app.command()
def extract_dataset(
    train_input_path: Path = RAW_DATA_DIR / "train.zip",
    test_input_path: Path = RAW_DATA_DIR / "test.zip",
    output_train_path: Path = PROCESSED_DATA_DIR / "train",
    output_test_path: Path = PROCESSED_DATA_DIR / "test"
):
    extract_zip_files(train_input_path, output_train_path)
    extract_zip_files(test_input_path, output_test_path)

@app.command()
def make_dataset(train_input_path: Path = PROCESSED_DATA_DIR / "train",
    output_path: Path = INTERIM_DATA_DIR,
):
    raw_df = extract_raw_data(train_input_path/'train.csv')
    test_size, random_state = read_params('params.yaml')
    train_df, val_df = train_val_split(raw_df,test_size,random_state)
    save_data(train_df, output_path/'train.csv')
    save_data(val_df,output_path/'val.csv')

    
if __name__ == "__main__":
    app()
