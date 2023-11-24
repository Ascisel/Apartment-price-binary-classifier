import argparse
from src.tools import load_dataset, process_df, get_dataloader, get_tensor_dataset, split_train_df
from src.config import TrainConfig as Config
from src.train_model import train_model, validate
from src.models import RealEstateClassifier
from datetime import datetime
import torch


parser = argparse.ArgumentParser(description='CLI wrapper for Apartment price classificator')
parser.add_argument('--train-path', default='data/train_data.csv',
                    help='path for the train dataset')

args = parser.parse_args()

train_path = args.train_path

def training(train_filename: str, device: str):
    current_datetime = datetime.now()

    df = load_dataset(train_filename)
    
    num_df, cat_df = process_df(df)

    train_num_df, train_cat_df, val_num_df, val_cat_df = split_train_df(num_df, cat_df, Config.VALIDATION_DATASET_RATE)

    n_num_columns, n_cat_columns = train_num_df.shape[1], train_cat_df.shape[1]
    # We take out target column from out numeric arguments while making dataset
    n_num_columns -= 1

    train_dataset = get_tensor_dataset(train_num_df, train_cat_df)
    val_dataset = get_tensor_dataset(val_num_df, val_cat_df)


    train_dataloader = get_dataloader(train_dataset, Config.BATCH_SIZE, True)
    val_dataloader = get_dataloader(val_dataset, Config.BATCH_SIZE, False)

    model = RealEstateClassifier(n_num_columns, n_cat_columns).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    loss_fn = torch.nn.BCEWithLogitsLoss()


    best_acc = 0

    for epoch in range(Config.EPOCHS):
        train_model(model, train_dataloader, loss_fn,
                    epoch, optimizer, device)
        
        acc = validate(model, val_dataloader, device)
    
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        if is_best:
            torch.save(model.state_dict(), f'models/best_model_{n_num_columns}_{n_cat_columns}_{current_datetime.year}-{current_datetime.month}-{current_datetime.day}.pt')


    best_model = RealEstateClassifier(n_num_columns, n_cat_columns).to(device)
    best_model.load_state_dict(torch.load(f'models/best_model_{n_num_columns}_{n_cat_columns}_{current_datetime.year}-{current_datetime.month}-{current_datetime.day}.pt'))
    acc = validate(best_model, val_dataloader, device)

    print(f'Best model accuracy: {acc:.3}')

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    training(train_path, device)  
