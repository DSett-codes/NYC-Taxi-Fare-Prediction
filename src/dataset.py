from pathlib import Path
from zipfile import ZipFile
from loguru import logger
from tqdm import tqdm
import typer

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

def extract_zip_files(input_path: Path,output_path: Path):
    with ZipFile(file=input_path) as f:
        f.extractall(path=output_path)
        logger.info(f"{input_path} extracted successfully at the target path")

@app.command()
def main(
    train_input_path: Path = RAW_DATA_DIR / "train.zip",
    test_input_path: Path = RAW_DATA_DIR / "test.zip",
    output_train_path: Path = PROCESSED_DATA_DIR / "train",
    output_test_path: Path = PROCESSED_DATA_DIR / "test"
):
    extract_zip_files(train_input_path, output_train_path)
    extract_zip_files(test_input_path, output_test_path)
    # # ---- REPLACE THIS WITH YOUR OWN CODE ----
    # logger.info("Processing dataset...")
    # for i in tqdm(range(10), total=10):
    #     if i == 5:
    #         logger.info("Something happened for iteration 5.")
    # logger.success("Processing dataset complete.")
    # # -----------------------------------------


if __name__ == "__main__":
    app()
