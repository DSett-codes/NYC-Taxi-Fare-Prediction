from pathlib import Path
import joblib
import sys
import pandas as pd
from yaml import safe_load
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

from loguru import logger
from tqdm import tqdm
import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

TARGET = 'trip_duration'

def load_dataframe(path):
    df = pd.read_csv(path)
    return df
    
    
    
def make_X_y(dataframe:pd.DataFrame,target_column:str):
    df_copy = dataframe.copy()
    
    X = df_copy.drop(columns=target_column)
    y = df_copy[target_column]
    
    return X, y


def train_model(model,X_train,y_train):
    # fit the model on data
    model.fit(X_train,y_train)
    logger.success('Model Training Complete')
    
    return model


def save_model(model,save_path):
    joblib.dump(value=model,
                filename=save_path)
    
    
@app.command()
def main(
    training_data_path: Path= PROCESSED_DATA_DIR / 'final' / 'train.csv',
    model_output_path: Path = MODELS_DIR / 'models'
):
 
    train_data = load_dataframe(training_data_path)
    # split the data into X and y
    X_train, y_train = make_X_y(dataframe=train_data,target_column=TARGET)
    # read the parameters from params.yaml
    with open('params.yaml') as f:
        params = safe_load(f)
    # make the model object
    regressor = XGBRegressor(**params['train_model']['xgb_regressor'])
    # train the model
    regressor = train_model(model=regressor,
                            X_train=X_train,
                            y_train=y_train)
    # save the model after training
   
    model_output_path.mkdir(exist_ok=True)
    save_model(model=regressor,save_path=model_output_path / 'xgbreg.joblib')


if __name__ == "__main__":
    app()
