from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PowerTransformer, StandardScaler
from yaml import safe_load

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.utils import OutliersRemover, euclidean_distance, haversine_distance, manhattan_distance

app = typer.Typer()

TARGET_COLUMN = "trip_duration"
PLOT_PATH = Path("reports/figures/target_distribution.png")

DISTANCE_FEATURES = {
    "haversine_distance": haversine_distance,
    "euclidean_distance": euclidean_distance,
    "manhattan_distance": manhattan_distance,
}

COORDINATE_COLUMNS = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]


def read_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_data(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def drop_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        col
        for col in ["id", "dropoff_datetime", "store_and_fwd_flag"]
        if col in dataframe.columns
    ]
    transformed = dataframe.drop(columns=columns_to_drop)
    logger.info(f"Dropped columns: {columns_to_drop}")
    return transformed


def make_datetime_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "pickup_datetime" not in dataframe.columns:
        return dataframe

    transformed = dataframe.copy()
    pickup_time = pd.to_datetime(transformed["pickup_datetime"])
    transformed["pickup_hour"] = pickup_time.dt.hour
    transformed["pickup_date"] = pickup_time.dt.day
    transformed["pickup_month"] = pickup_time.dt.month
    transformed["pickup_day"] = pickup_time.dt.weekday
    transformed["is_weekend"] = (transformed["pickup_day"] >= 5).astype(int)

    transformed = transformed.drop(columns=["pickup_datetime"])
    return transformed


def remove_passengers(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "passenger_count" not in dataframe.columns:
        return dataframe
    return dataframe.loc[dataframe["passenger_count"].isin(range(1, 7)), :].copy()


def convert_target_to_minutes(dataframe: pd.DataFrame, target_column: str) -> pd.DataFrame:
    transformed = dataframe.copy()
    transformed[target_column] = transformed[target_column] / 60.0
    return transformed


def drop_above_two_hundred_minutes(dataframe: pd.DataFrame, target_column: str) -> pd.DataFrame:
    return dataframe.loc[dataframe[target_column] <= 200, :].copy()


def plot_target(dataframe: pd.DataFrame, target_column: str, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    sns.kdeplot(data=dataframe, x=target_column)
    plt.title(f"Distribution of {target_column}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def input_modifications(dataframe: pd.DataFrame) -> pd.DataFrame:
    transformed = drop_columns(dataframe)
    transformed = remove_passengers(transformed)
    transformed = make_datetime_features(transformed)
    logger.info("Input feature modifications complete")
    return transformed


def target_modifications(dataframe: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    transformed = convert_target_to_minutes(dataframe, target_column)
    transformed = drop_above_two_hundred_minutes(transformed, target_column)
    plot_target(transformed, target_column, PLOT_PATH)
    logger.info("Target modifications complete")
    return transformed


def modify_features(data_path: Path) -> pd.DataFrame:
    dataframe = read_data(data_path)
    transformed = input_modifications(dataframe)
    if TARGET_COLUMN in transformed.columns:
        transformed = target_modifications(transformed)
    return transformed


def implement_distances(dataframe: pd.DataFrame) -> pd.DataFrame:
    transformed = dataframe.copy()
    lat1 = transformed["pickup_latitude"]
    lon1 = transformed["pickup_longitude"]
    lat2 = transformed["dropoff_latitude"]
    lon2 = transformed["dropoff_longitude"]

    for feature_name, func in DISTANCE_FEATURES.items():
        transformed[feature_name] = func(lat1, lon1, lat2, lon2)

    return transformed


def remove_outliers(dataframe: pd.DataFrame, percentiles: list[float], column_names: list[str]) -> OutliersRemover:
    transformer = OutliersRemover(percentile_values=percentiles, col_subset=column_names)
    transformer.fit(dataframe)
    return transformer


def train_preprocessor(data: pd.DataFrame) -> ColumnTransformer:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "one-hot",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                ["vendor_id"],
            ),
            ("min-max", MinMaxScaler(), COORDINATE_COLUMNS),
            (
                "standard-scale",
                StandardScaler(),
                ["haversine_distance", "euclidean_distance", "manhattan_distance"],
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
        n_jobs=1,
    )
    preprocessor.set_output(transform="pandas")
    preprocessor.fit(data)
    return preprocessor


def transform_data(transformer, data):
    return transformer.transform(data)


def transform_output(target: pd.Series) -> PowerTransformer:
    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    transformer.fit(target.to_numpy().reshape(-1, 1))
    return transformer


def save_transformer(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(value=obj, filename=path)


def iter_csv_inputs(input_paths: list[Path]) -> list[Path]:
    resolved_paths: list[Path] = []
    for path in input_paths:
        if path.is_dir():
            resolved_paths.extend(sorted(path.glob("*.csv")))
        elif path.suffix == ".csv":
            resolved_paths.append(path)
    return resolved_paths


@app.command()
def modify(
    input_paths: list[Path],
    output_path: Path = typer.Option(PROCESSED_DATA_DIR / "transformations", "--output-path"),
):
    csv_inputs = iter_csv_inputs(input_paths)
    if not csv_inputs:
        raise typer.BadParameter("No CSV input files were provided.")

    for path in csv_inputs:
        transformed = modify_features(path)
        save_path = output_path / path.name
        save_data(transformed, save_path)
        logger.info(f"Saved transformed file to: {save_path}")


@app.command()
def build(
    input_paths: list[Path],
    output_path: Path = typer.Option(PROCESSED_DATA_DIR / "build-features", "--output-path"),
):
    csv_inputs = iter_csv_inputs(input_paths)
    if not csv_inputs:
        raise typer.BadParameter("No CSV input files were provided.")

    for path in csv_inputs:
        dataframe = read_data(path)
        transformed = implement_distances(dataframe)
        save_path = output_path / path.name
        save_data(transformed, save_path)
        logger.info(f"Saved built-features file to: {save_path}")


@app.command("preprocess")
def preprocess(
    filenames: list[str] = typer.Argument(["train.csv", "val.csv", "test.csv"]),
    input_path: Path = typer.Option(PROCESSED_DATA_DIR / "build-features", "--input-path"),
    output_path: Path = typer.Option(PROCESSED_DATA_DIR / "final", "--output-path"),
    transformers_path: Path = typer.Option(MODELS_DIR / "transformers", "--transformers-path"),
    params_path: Path = typer.Option(Path("params.yaml"), "--params-path"),
):
    with open(params_path, encoding="utf-8") as stream:
        params = safe_load(stream)

    percentiles = list(params["data_preprocessing"]["percentiles"])

    for filename in filenames:
        data_file = input_path / filename
        dataframe = read_data(data_file)

        if filename == "train.csv":
            X = dataframe.drop(columns=TARGET_COLUMN)
            y = dataframe[TARGET_COLUMN]

            outlier_transformer = remove_outliers(X, percentiles, COORDINATE_COLUMNS)
            X_no_outliers = transform_data(outlier_transformer, X)
            y_no_outliers = y.loc[X_no_outliers.index]
            save_transformer(transformers_path / "outliers.joblib", outlier_transformer)

            preprocessor = train_preprocessor(X_no_outliers)
            X_transformed = transform_data(preprocessor, X_no_outliers)
            save_transformer(transformers_path / "preprocessor.joblib", preprocessor)

            output_transformer = transform_output(y_no_outliers)
            y_transformed = transform_data(output_transformer, y_no_outliers.to_numpy().reshape(-1, 1)).ravel()
            save_transformer(transformers_path / "output_transformer.joblib", output_transformer)

            X_transformed[TARGET_COLUMN] = y_transformed
            save_data(X_transformed, output_path / filename)

        elif filename == "val.csv":
            X = dataframe.drop(columns=TARGET_COLUMN)
            y = dataframe[TARGET_COLUMN]

            outlier_transformer = joblib.load(transformers_path / "outliers.joblib")
            preprocessor = joblib.load(transformers_path / "preprocessor.joblib")
            output_transformer = joblib.load(transformers_path / "output_transformer.joblib")

            X_no_outliers = transform_data(outlier_transformer, X)
            y_no_outliers = y.loc[X_no_outliers.index]
            X_transformed = transform_data(preprocessor, X_no_outliers)
            y_transformed = transform_data(output_transformer, y_no_outliers.to_numpy().reshape(-1, 1)).ravel()

            X_transformed[TARGET_COLUMN] = y_transformed
            save_data(X_transformed, output_path / filename)

        elif filename == "test.csv":
            outlier_transformer = joblib.load(transformers_path / "outliers.joblib")
            preprocessor = joblib.load(transformers_path / "preprocessor.joblib")

            X_no_outliers = transform_data(outlier_transformer, dataframe)
            X_transformed = transform_data(preprocessor, X_no_outliers)
            save_data(X_transformed, output_path / filename)

        else:
            logger.warning(f"Skipping unsupported filename: {filename}")

        logger.info(f"Preprocessed file: {filename}")


if __name__ == "__main__":
    app()
