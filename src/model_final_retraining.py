from models import *
from algorithms import *

import pandas
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
import wandb
from torch.utils.data import DataLoader
import os
import numpy as np
import time
import tqdm
from sklearn.model_selection import train_test_split


os.environ['WANDB_NOTEBOOK_NAME'] = 'model_training.ipynb'
wandb.login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#print('device in use:', torch.cuda.get_device_name(device))

training_set = pandas.read_csv("../data/merged_dataset_proc.csv", sep=";").sample(frac=1, random_state=69)

training_len = len(training_set)

max_token_len = 128

#model_name = 'FacebookAI/xlm-roberta-base'
# model_name = '/opt/models/FacebookAI/xlm-roberta-base'

verbose = True

# class to create a custom "DataSet" object, compatible with pytorch
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer):
        self.len = len(dataframe)

        # tokenized dataframe
        self.input_values = [tokenizer(a, padding="max_length", max_length=max_token_len,  return_tensors='pt', truncation=True) for a in dataframe["processed_tweet"].values]
        # gold labels
        self.labels = torch.from_numpy(dataframe[['NO', 'IDEOLOGICAL-INEQUALITY', 'STEREOTYPING-DOMINANCE',
                                                  'OBJECTIFICATION', 'SEXUAL-VIOLENCE', 'MISOGYNY-NON-SEXUAL-VIOLENCE']].values.astype(np.float32))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        input_values = self.input_values[idx]
        labels = self.labels[idx]

        return input_values, labels

# correct tokenizer releted to the selected pretrained transformer
#tokenizer = AutoTokenizer.from_pretrained(model_name)

# training and test sets are refactored as "CustomDataset" objects
#tr_data = CustomDataset(training_set, tokenizer=tokenizer)
#val_data = CustomDataset(test_set, tokenizer=tokenizer)

# function to call to save the model in a specific path, file name must be .pt
def save_model(model:nn.Module, path:str):
    torch.save(model.state_dict(), path)

# load a saved model (MultiLabelClassifier class)
# config necessary to reproduce the same nn architecture (pretrained model, hidden size, output size ecc...)
def load_model(path:str, config:dict): 
    if config['class'] == 'not_chain':
        classifier = MultiLabelClassifier(config['pretrained_name'], config['hidden_layer_size'])
    else:
        classifier = MultiLabelClassifierChain(config['pretrained_name'], config['hidden_layer_size'])
    
    classifier.load_state_dict(torch.load(path))

    return classifier

def my_collate_fn(batch):

    # tokenized samples (inputs and targets) are grouped in batches
    input = {'input_ids':torch.stack(([x[0]['input_ids'][0] for x in batch])).to(device), 'attention_mask':torch.stack(([x[0]['attention_mask'][0] for x in batch])).to(device)}
    labels = torch.stack(([x[1] for x in batch])).to(device)

    return [input, labels]

def train_one_epoch(epoch, tot_batch, tr_data_loader, n_split, classifier, optimizer, criterion):
    tr_loss = 0.0
    epoch_time_start = time.time()

    # reset of gradients
    optimizer.zero_grad()

    with tqdm.tqdm(total=tot_batch, desc=f'epoch {epoch}') as pbar:

        batch_loss = 0
        for i, batch in enumerate(tr_data_loader):

            input, labels = batch

            # setting the timer for calculating statistics
            batch_time_start = time.time()

            # Forward Phase
            if isinstance(classifier, MultiLabelClassifierChain):
                output = classifier(input, labels[:,:1])
            else:
                output = classifier(input)

            loss = criterion(output, labels)
            tr_loss += loss.item()
            
            loss = loss/n_split
            
            batch_loss += loss.item()
            

            # Backward Phase - Gradient Accumulation
            loss.backward()

            # Whenever the gradient of a real mini-batch of data is accumulated, a learning step is performed
            if (i + 1)%n_split == 0:

                optimizer.step()

                # reset of gradients
                optimizer.zero_grad()

                batch_time = time.time() - batch_time_start
                wandb.log({'tr_batch_loss': batch_loss, 'batch_time':batch_time})
                batch_loss = 0

                pbar.update(1)

    return tr_loss/(i + 1), time.time() - epoch_time_start

def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    elif optimizer == "adamw":
        optimizer = optim.AdamW(network.parameters(),
                               lr=learning_rate)
    return optimizer

def train(config=None):
    
    class Configuration:
        def __init__(self, learning_rate, batch_size, epochs, pretrained_name, optimizer, hidden_layer_size, freeze_pretrained, dropout, classifier_type, path_name):
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.epochs = epochs
            self.optimizer = optimizer
            self.hidden_layer_size = hidden_layer_size
            self.pretrained_name = pretrained_name
            self.freeze_pretrained = freeze_pretrained
            self.dropout = dropout
            self.classifier_type = classifier_type
            self.path_name = path_name
            
    config_best_icm_soft = Configuration(learning_rate = 0.000036936026,
                                         batch_size = 64,
                                         epochs = 4,
                                         pretrained_name = 'sdadas/xlm-roberta-large-twitter',
                                         optimizer = 'adamw',
                                         hidden_layer_size = 512,
                                         freeze_pretrained = False,
                                         dropout = 0.4,
                                         classifier_type = 'ff',
                                         path_name = '../data/models/best_icm_soft.pt')
    
    config_best_chain_val = Configuration(learning_rate = 0.00001,
                                         batch_size = 64,
                                         epochs = 7,
                                         pretrained_name = 'sdadas/xlm-roberta-large-twitter',
                                         optimizer = 'adamw',
                                         hidden_layer_size = 128,
                                         freeze_pretrained = False,
                                         dropout = 0.2,
                                         classifier_type = 'chain',
                                         path_name = '../data/models/best_chain_val.pt')
    
    config_best_ff_val = Configuration(  learning_rate = 0.000018176664,
                                         batch_size = 64,
                                         epochs = 4,
                                         pretrained_name = 'sdadas/xlm-roberta-large-twitter',
                                         optimizer = 'adamw',
                                         hidden_layer_size = 512,
                                         freeze_pretrained = False,
                                         dropout = 0.25,
                                         classifier_type = 'ff',
                                         path_name = '../data/models/best_ff_val.pt')
        
    with wandb.init(config=config):

        if wandb.config.model_name == 'best_icm_soft':
            config = config_best_icm_soft
        elif wandb.config.model_name == 'best_chain_val': 
            config = config_best_chain_val
        elif wandb.config.model_name == 'best_ff_val':  
            config = config_best_ff_val
            
        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_name)

        tr_data = CustomDataset(training_set, tokenizer=tokenizer)

        mean_epoch_time = 0
        tot_batch = int(training_len/config.batch_size)

        # maximum batch size supportable by the GPU of the current device
        # !!!!!!!!!! supportable_batch_size MUST BE A DIVISOR OF config.batch_size !!!!!!!!!!
        supportable_batch_size = 64

        supportable_batch_size = min(supportable_batch_size, config.batch_size)
        n_split = int(config.batch_size/supportable_batch_size)

        if verbose: print(n_split, "shots of", supportable_batch_size,
              "elements  -  config batch_size:", config.batch_size)

        if 'large' in config.pretrained_name:
            transf_out_size = 1024
        else:
            transf_out_size = 768

        seed = int(time.time())
        torch.manual_seed(seed)
        wandb.log({'seed':seed})
        
        if config.classifier_type == 'chain':
            classifier = MultiLabelClassifierChain(config.pretrained_name,
                                                   config.dropout,
                                                   config.hidden_layer_size,
                                                   int(config.hidden_layer_size*(1/2)),
                                                   transf_out_size
                                                   ).to(device)
        else:
            classifier = MultiLabelClassifier(config.pretrained_name,
                                              config.dropout,
                                              config.hidden_layer_size,
                                              int(config.hidden_layer_size*(1/2)),
                                              transf_out_size
                                              ).to(device)
        if config.freeze_pretrained:
            print("freeze")
            classifier.freeze_pretrained()

        criterion = nn.BCELoss()
        optimizer = build_optimizer(classifier, config.optimizer, config.learning_rate)

        # "DataLoader" objects are exploited to iterate on the dataset in batches
        tr_data_loader = DataLoader(tr_data, batch_size=supportable_batch_size, shuffle=True, collate_fn=my_collate_fn)

        for epoch in range(config.epochs):

            # model modules switch their behaviour to "training mode"
            classifier.train()
            tr_loss, epoch_time = train_one_epoch(epoch, tot_batch, tr_data_loader, n_split, classifier, optimizer, criterion)
            mean_epoch_time += epoch_time
            
        mean_epoch_time /= epoch + 1

        wandb.log({'mean_epoch_time':mean_epoch_time})
        
        save_model(classifier, config.path_name)

wandb.agent('hltproject/EXIST2024/ks47deee', train)
wandb.finish()
    
