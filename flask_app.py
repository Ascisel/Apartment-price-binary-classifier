import pandas as pd
import argparse
from flask import Flask, request
from src.models import RealEstateClassifier
from src.tools import get_num_columns, get_model_name, process_df, get_tensor_dataset
from src.config import TaskConfig
import torch
app = Flask(__name__)

global model
global device

def process_data(data: dict):
    input_dataframe = pd.DataFrame(data)
    num_df, cat_df = process_df(input_dataframe)

    return num_df, cat_df

@app.route('/predict, method="POST"')
def predict():
    data = request.get_json(force=True)
    model.to(device)
    num_df, cat_df = process_data(data)
    num_input = torch.from_numpy(num_df.values).float().to(device)
    cat_input = torch.from_numpy(cat_df.values).float().to(device)
    preds = model(num_input, cat_input)
    return preds.tolist()




@app.route('/health', methods=['GET'])
def health():
    if model is not None:
        return "OK"
    return "Model is not initialized"


def parse_args():
    parser = argparse.ArgumentParser(description='Apartment Price model API')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--ip', type=str, default='0.0.0.0')

if __name__ == '__main__':

    args = parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_name = get_model_name()
    n_num_columns, n_cat_columns = get_num_columns(model_name)
    model = RealEstateClassifier(n_num_columns, n_cat_columns).to(device)
    model.load_state_dict(torch.load(TaskConfig.MODELS_DIR + '/' + model_name))
    
    app.run(ip=args.ip, port=args.port)



