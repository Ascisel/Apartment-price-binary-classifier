import pandas as pd
import numpy as np
import torch
import re
import os
from typing import List, Tuple
from datetime import datetime
from src.config import TaskConfig as Config



def convert_time(column: str, df: pd.DataFrame) -> pd.DataFrame:
    try:
        for time in df[column].unique():
            if time == 'no_bus_stop_nearby':
                df[column] = df[column].replace(time, '60')
            else:
                df[column] = df[column].replace(time, str(re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", time)[-1]))
        df[column] = df[column].astype(float, errors='raise').abs()
        return df
    except Exception:
        print('Data already converted')
        return df


def change_cat_values_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    non_numeric_df = df.select_dtypes('object')
    unique_values = {column : non_numeric_df[column].unique() for column in non_numeric_df.columns}
    binary_cat_columns = [column for column in unique_values.keys() if len(unique_values[column]) == 2]
    
    for column in binary_cat_columns:
        df[column] = (df[column] == unique_values[column][0]).astype(float)

    return df


def change_goal_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    df.SalePrice = (df.SalePrice > Config.PRIZE_THRESHOLD).astype(float)

    return df


def get_dataloader(
    dataset: torch.utils.data.dataset.TensorDataset,
    batch_size: int,
    shuffle: bool
) -> torch.utils.data.dataloader.DataLoader:
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle)
    return dataloader


def get_tensor_dataset(
    num_df: pd.DataFrame,
    cat_df: pd.DataFrame,
) -> torch.utils.data.dataloader.DataLoader:
    
    num_data, cat_data, target_data = process_to_tensors(num_df, cat_df)

    tensor_dataset = torch.utils.data.TensorDataset(num_data, cat_data, target_data)

    return tensor_dataset


def load_dataset(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)

    return df

def get_one_hot_cat_columns(df: pd.DataFrame, cat_columns: List[str]) -> pd.DataFrame:
    categorical_train_values = pd.get_dummies(df[cat_columns])

    return categorical_train_values


def get_train_indices(num_df_rows: int, validation_rate: float) -> np.ndarray:
    train_indices = np.random.rand(num_df_rows) > validation_rate

    return train_indices


def process_to_tensors(
    num_df: pd.DataFrame,
    cat_df: pd.DataFrame
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    numerical_tensor_data = torch.from_numpy(num_df.values[:, 1:]).float()
    categorical_tensor_data = torch.from_numpy(cat_df.values).float()
    tensor_targets = torch.from_numpy(num_df.values[:, 0]).float()

    return numerical_tensor_data, categorical_tensor_data, tensor_targets


def split_train_df(
    num_df: pd.DataFrame,
    cat_df:pd.DataFrame,
    validation_rate: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    train_indices = get_train_indices(len(num_df), validation_rate)

    train_cat_df = cat_df.iloc[train_indices]
    train_num_df = num_df.iloc[train_indices]

    val_cat_df = cat_df.iloc[~train_indices]
    val_num_df = num_df.iloc[~train_indices]
    
    return train_num_df, train_cat_df, val_num_df, val_cat_df

def process_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = change_goal_to_binary(df)
    df = change_cat_values_to_binary(df)

    time_col = ['TimeToBusStop', 'TimeToSubway']
    for col in time_col:
        df = convert_time(col, df)

    cat_columns = ['HallwayType', 'SubwayStation']
    cat_df = get_one_hot_cat_columns(df, cat_columns)
    num_df = df.drop(columns=cat_columns)


    return num_df, cat_df

def get_model_name() -> str:
    files = os.listdir(os.path.abspath(Config.MODELS_DIR))

    # Initialize variables to store the oldest date and corresponding filename
    oldest_date = datetime.max
    oldest_filename = None

    # Loop through the files and find the one with the oldest date
    for file in files:
        try:
            # Extract the date from the filename
            date_str = file.split('_')[-1].split('.')[0]
            file_date = datetime.strptime(date_str, '%Y-%m-%d')

            # Check if this date is older than the current oldest date
            if file_date < oldest_date:
                oldest_date = file_date
                oldest_filename = file
        except ValueError as e:
            # Ignore files with date parsing errors
            print('Value error occured while trying to parse date from filename', e)
            pass

    return oldest_filename

def get_num_columns(model_name: str) -> Tuple[int, int]:
    pattern = r'best_model_(\d+)_(\d+)_'

    # Use re.match to find the matching pattern
    match = re.match(pattern, model_name)

    # Check if the pattern is found
    if match:
    # Extract the values using group() method
        value1 = int(match.group(1))
        value2 = int(match.group(2))
    return value1, value2

if __name__ == '__main__':
    df = load_dataset(r'C:\Users\alber\Desktop\nudes\Apartment-price-binary-classifier\data\train_data.csv')
    
    train_dataset, val_dataset = process_df(df)
