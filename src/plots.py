from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info(f"Reading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    logger.info("Generating plot from data...")
    plt.figure(figsize=(10, 6))
    df.plot()
    plt.savefig(output_path)
    logger.success(f"Plot generation complete. Saved to {output_path}.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
