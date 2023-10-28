import argparse
from src.tools import load_dataset, process_df, get_dataloader, get_tensor_dataset
from src.config import TrainConfig as Config
from src.train_model import train_model
from src.models import RealEstateClassifier
import torch

parser = argparse.ArgumentParser(description='CLI wrapper for Apartment price classificator')
parser.add_argument('--train-path', default='data/train_data.csv',
                    help='path for the train dataset')

args = parser.parse_args()

train_path = args.train_path

def training(train_filename: str, device: str):
    df = load_dataset(train_filename)
    
    train_num_df, train_cat_df, val_num_df, val_cat_df = process_df(df)

    n_num_columns, n_cat_columns = train_num_df.shape[1], train_cat_df.shape[1]

    train_dataset = get_tensor_dataset(train_num_df, train_cat_df)
    val_dataset = get_tensor_dataset(val_num_df, val_cat_df)

    # We take out target column from out numeric arguments while making dataset
    n_num_columns -= 1

    train_dataloader = get_dataloader(train_dataset, Config.BATCH_SIZE, True)
    val_dataloader = get_dataloader(val_dataset, Config.BATCH_SIZE, False)

    model = RealEstateClassifier(n_num_columns, n_cat_columns).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model = train_model(model, train_dataloader, val_dataloader, loss_fn,
                        Config.EPOCHS, optimizer, device)
    
    

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    training(train_path, device)
