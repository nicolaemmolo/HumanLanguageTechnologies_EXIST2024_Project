import pandas
import ast
from algorithms import *
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import transformers
from models import *
from transformers import AutoModel, AutoTokenizer
import wandb
from torch.utils.data import DataLoader
import os
import time
import pathlib
import numpy as np
import time
import json
from os import listdir
from os.path import isfile, join
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils
import json
import pathlib


config_best_icm_soft = {
    'pretrained_name' : 'sdadas/xlm-roberta-large-twitter',
    'max_token_length': 128,
    'batch_size' : 64,
    'dropout': 0.4,
    'hidden_layer_1': 512,
    'hidden_layer_2': 256,
    'transf_out': 1024,
    'class': 'ff'
}

config_best_chain_val = {
    'pretrained_name' : 'sdadas/xlm-roberta-large-twitter',
    'max_token_length': 128,
    'batch_size' : 64,
    'dropout': 0.2,
    'hidden_layer_1': 128,
    'hidden_layer_2': 64,
    'transf_out': 1024,
    'class': 'chain'
}

config_best_ff_val = {
    'pretrained_name' : 'sdadas/xlm-roberta-large-twitter',
    'max_token_length': 128,
    'batch_size' : 64,
    'dropout': 0.25,
    'hidden_layer_1': 512,
    'hidden_layer_2': 256,
    'transf_out': 1024,
    'class': 'ff'
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path:str, hyperparam:dict): 
    if hyperparam['class'] == 'ff':
        classifier = MultiLabelClassifier(hyperparam['pretrained_name'], hyperparam['dropout'], 
                                          hyperparam['hidden_layer_1'], hyperparam['hidden_layer_2'], hyperparam['transf_out'])
    else:
        classifier = MultiLabelClassifierChain(hyperparam['pretrained_name'], hyperparam['dropout'], 
                                               hyperparam['hidden_layer_1'], hyperparam['hidden_layer_2'], hyperparam['transf_out'])
    
    classifier.load_state_dict(torch.load(path))

    return classifier


# class to create the DataSet object, as required by pytorch
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, max_token_len):
        self.len = len(dataframe)

        # tokenized dataframe
        self.input_values = [tokenizer(a, padding="max_length", max_length=max_token_len,  return_tensors='pt', truncation=True) for a in dataframe["processed_tweet"].values]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        input_values = self.input_values[idx]

        return input_values


def test_collate_fn(batch):
    # tokenized samples (inputs and targets) are grouped in batches
    input = {'input_ids':torch.stack(([x['input_ids'][0] for x in batch])).to(device), 'attention_mask':torch.stack(([x['attention_mask'][0] for x in batch])).to(device)}
        
    return input


def model_json_output(model_path:str, hyperparam:dict, test_df_path:str, json_path:str = 'task3_soft_Medusa_1.json', categories:list = ['NO', 'IDEOLOGICAL-INEQUALITY', 'STEREOTYPING-DOMINANCE', 
                                                                                            'OBJECTIFICATION', 'SEXUAL-VIOLENCE', 'MISOGYNY-NON-SEXUAL-VIOLENCE']):
    model = load_model(model_path, hyperparam)
    model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(hyperparam['pretrained_name'])
    test_df = pandas.read_csv(test_df_path, sep=";")

    ids = test_df['id_EXIST'].to_list() # ids necessary for the json format of the challenge
    test_data = TestDataset(test_df, tokenizer, hyperparam['max_token_length'])
    test_data_loader = DataLoader(test_data, batch_size=hyperparam['batch_size'], shuffle=False, collate_fn=test_collate_fn)

    json_data = [] # list of dict

    model.eval()
    with torch.no_grad():
        for ts_input in test_data_loader:
            tensor_output = model(ts_input)

            for output in tensor_output: # iterate over the batch output
                soft_pred = {}

                for i, category in enumerate(categories):
                    soft_pred[category] = output[i].item() # save soft pred in a dict (the categories are the keys)

                json_data.append({"test_case": "EXIST2024", "id": str(ids.pop(0)), "value": soft_pred})
    
    with open(json_path, 'w') as json_file: # save all dicts in the json file
        json.dump(json_data, json_file, indent=1)



path_name_best_icm_soft = '../data/models/best_icm_soft.pt'

path_name_best_chain_val = '../data/models/best_chain_val.pt'

path_name_best_ff_val = '../data/models/best_ff_val.pt'


model_json_output(model_path=path_name_best_icm_soft, hyperparam=config_best_icm_soft, test_df_path='../data/real_test_proc.csv', json_path='task3_soft_Medusa_1.json')
model_json_output(model_path=path_name_best_chain_val, hyperparam=config_best_chain_val, test_df_path='../data/real_test_proc.csv', json_path='task3_soft_Medusa_2.json')
model_json_output(model_path=path_name_best_ff_val, hyperparam=config_best_ff_val, test_df_path='../data/real_test_proc.csv', json_path='task3_soft_Medusa_3.json')
