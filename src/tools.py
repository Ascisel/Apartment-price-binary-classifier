import pandas as pd
import numpy as np
import torch
import re
from typing import List, Tuple
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
    binary_cat_columns = [column for column in unique_values.keys()
                           if len(unique_values[column]) == 2]
    
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


def get_train_indices(num_df_rows: int) -> np.ndarray:
    train_indices = np.random.rand(num_df_rows) > Config.VALIDATION_DATASET_RATE

    return train_indices


def process_to_tensors(
    num_df: pd.DataFrame,
    cat_df: pd.DataFrame
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    numerical_tensor_data = torch.from_numpy(num_df.values[:, 1:]).float()
    categorical_tensor_data = torch.from_numpy(cat_df.values).float()
    tensor_targets = torch.from_numpy(num_df.values[:, 0]).float()

    return numerical_tensor_data, categorical_tensor_data, tensor_targets


def process_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = change_goal_to_binary(df)
    df = change_cat_values_to_binary(df)

    time_col = ['TimeToBusStop', 'TimeToSubway']
    for col in time_col:
        df = convert_time(col, df)

    cat_columns = ['HallwayType', 'SubwayStation']
    cat_df = get_one_hot_cat_columns(df, cat_columns)
    num_df = df.drop(columns=cat_columns)
    train_indices = get_train_indices(len(df))

    train_cat_df = cat_df.iloc[train_indices]
    train_num_df = num_df.iloc[train_indices]

    val_cat_df = cat_df.iloc[~train_indices]
    val_num_df = num_df.iloc[~train_indices]

    return train_num_df, train_cat_df, val_num_df, val_cat_df


    # train_num_data, train_cat_data, train_targets = process_to_tensors(train_num_df, train_cat_df)
    # val_num_data, val_cat_data, val_targets = process_to_tensors(val_num_df, val_cat_df)
    
    # train_dataset = get_tensor_dataset(train_num_data, train_cat_data, train_targets)
    # val_dataset = get_tensor_dataset(val_num_data, val_cat_data, val_targets)

    # return train_dataset, val_dataset


if __name__ == '__main__':
    df = load_dataset(r'C:\Users\alber\Desktop\nudes\Apartment-price-binary-classifier\data\train_data.csv')
    
    train_dataset, val_dataset = process_df(df)
