from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import sys
import joblib
import pandas as pd
from sklearn.metrics import r2_score
from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

TARGET = 'trip_duration'
model_name = 'xgbreg.joblib'

def load_dataframe(path):
    df = pd.read_csv(path)
    return df
    
    
def make_X_y(dataframe:pd.DataFrame,target_column:str):
    df_copy = dataframe.copy()
    
    X = df_copy.drop(columns=target_column)
    y = df_copy[target_column]
    
    return X, y

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    data_path: Path = PROCESSED_DATA_DIR / 'final',
    model_path: Path = MODELS_DIR / "models" / model_name,
    predictions_path: Path = PROCESSED_DATA_DIR / 'test' ,
    # -----------------------------------------
):
    data = load_dataframe(data_path/'val.csv')
    # split the data into X and y
    X_test, y_test = make_X_y(dataframe=data,target_column=TARGET)
    # load the model
    model = joblib.load(model_path)
    # get predictions from model
    y_pred = model.predict(X_test)
    # calcuate the r2 score
    score = r2_score(y_test,y_pred)
    
    logger.info(f'The score for dataset {data_path.stem} is {score}')
    test_data = load_dataframe(data_path/'test.csv')
    test_pred = model.predict(test_data)
    df = pd.DataFrame({TARGET: test_pred})
    df.to_csv(predictions_path/'test_predictions.csv', index_label="index")
    
    logger.success(f'Test predictions has been saved to {predictions_path}')


if __name__ == "__main__":
    app()
