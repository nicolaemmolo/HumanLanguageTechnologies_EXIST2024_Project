import re
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import json
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils


# TOKENIZATION

def tokenize(tweet, tokenizer):
    """
    Process the tweet with our custom requirements and tokenizes it using the given tokenizer.

    Args:
        tweet (str): The input tweet to be tokenized.
        tokenizer: The tokenizer object to be used for tokenization.

    Returns:
        list: The tokenized representation of the tweet.
    """
    
    # useful patterns and token for substitutions
    link_token = 'TURL'
    cit_token = 'TREF'
    number_tok = 'TNUMBER'
    multiple_question_marks = 'TMQM'
    multiple_exclamation_marks = 'TMEM'
    mixed_exclamation_question_marks = 'TMEQM'
    
    # substitution of urls
    tweet = re.sub(r'(https?://[^\s]+)', '', tweet)
    
    # substitution of tags
    tweet = re.sub(r'@[\w]+', '', tweet)
    
    # substitution of #blabla
    # tweet = re.sub(r'#([\w]+)', r' \1', tweet)
    
    # substituition of repetition of .
    tweet = re.sub(r"\.\.\.\.+", r'...', tweet)
    
    # the use of ¿ ¡ or is unreliable on social media, what to do?
    tweet = re.sub(r"¡", '', tweet)
    tweet = re.sub(r"¿", '', tweet)
    
    # substitution of multiple occurrence of ! ? and !?
    tweet = re.sub(r"(^|[^\?!])(!(\s*!)+)([^\?!]|$)", r'\1 ' + multiple_exclamation_marks + r' \4', tweet)
    tweet = re.sub(r"(^|[^\?!])(\?(\s*\?)+)([^\?!]|$)", r'\1 ' + multiple_question_marks + r' \4', tweet)
    tweet = re.sub(r"(^|[^\?!])([!\?](\s*[!\?])+)([^\?!]|$)", r'\1 ' + mixed_exclamation_question_marks + r' \4', tweet)
    
    tweet = re.sub(multiple_question_marks, '??', tweet)
    tweet = re.sub(multiple_exclamation_marks, '!!', tweet)
    tweet = re.sub(mixed_exclamation_question_marks, '?!', tweet)
    
    # substitution of numbers, dates ...
    tweet = re.sub(r'[-\+]?([0-9]+[\.:,;\\/ -])*[0-9]+', '', tweet) 
    
    # laugh
    # tweet = re.sub(r"\s[ja]{4,}", r'jaja', tweet)
    # laugh
    # tweet = re.sub(r"\s[ah]{4,}", r'haha', tweet)
    
    # add space after dots
    tweet = re.sub(r'([a-z])(\.|\.\.\.|\?|!|:|;|,|"|\)|}|]|…)(\w)', r'\1\2 \3', tweet)
    
    # remove all form of repetition
    tweet = re.sub(r"([^\.]+?)\1+", r'\1\1', tweet)
    
    # remove useless spaces
    tweet = re.sub(r"(\s)\1*", r'\1', tweet)
    tweet = re.sub(r"(^\s*|\s*$)", r'', tweet)
    
    # substitution of superfluous repetition in cha
    
    token = tokenizer.encode(tweet)
    
    return token

def process_tweet(tweet):
    """
    Process the tweet with our custom requirements and tokenizes it using the given tokenizer.

    Args:
        tweet (str): The input tweet to be tokenized.

    Returns:
        string: the processed tweet
    """
    
    # useful patterns and token for substitutions
    multiple_question_marks = 'TMQM'
    multiple_exclamation_marks = 'TMEM'
    mixed_exclamation_question_marks = 'TMEQM'
    
    # substitution of urls
    tweet = re.sub(r'(https?://[^\s]+)', '', tweet)
    
    # substitution of tags
    tweet = re.sub(r'@[\w]+', '', tweet)
    
    # substitution of #blabla
    # tweet = re.sub(r'#([\w]+)', r' \1', tweet)
    
    # substituition of repetition of .
    tweet = re.sub(r"\.\.\.\.+", r'...', tweet)
    
    # the use of ¿ ¡ or is unreliable on social media, what to do?
    tweet = re.sub(r"¡", '', tweet)
    tweet = re.sub(r"¿", '', tweet)
    
    # substitution of multiple occurrence of ! ? and !?
    tweet = re.sub(r"(^|[^\?!])(!(\s*!)+)([^\?!]|$)", r'\1 ' + multiple_exclamation_marks + r' \4', tweet)
    tweet = re.sub(r"(^|[^\?!])(\?(\s*\?)+)([^\?!]|$)", r'\1 ' + multiple_question_marks + r' \4', tweet)
    tweet = re.sub(r"(^|[^\?!])([!\?](\s*[!\?])+)([^\?!]|$)", r'\1 ' + mixed_exclamation_question_marks + r' \4', tweet)
    
    tweet = re.sub(multiple_question_marks, '??', tweet)
    tweet = re.sub(multiple_exclamation_marks, '!!', tweet)
    tweet = re.sub(mixed_exclamation_question_marks, '?!', tweet)
    
    # substitution of numbers, dates ...
    tweet = re.sub(r'[-\+]?([0-9]+[\.:,;\\/ -])*[0-9]+', '', tweet) 
    
    # add space after dots
    tweet = re.sub(r'([a-z])(\.|\.\.\.|\?|!|:|;|,|"|\)|}|]|…)(\w)', r'\1\2 \3', tweet)
    
    # remove all form of repetition
    tweet = re.sub(r"([^\.]+?)\1+", r'\1\1', tweet)
    
    # remove useless spaces
    tweet = re.sub(r"(\s)\1*", r'\1', tweet)
    tweet = re.sub(r"(^\s*|\s*$)", r'', tweet)
    
    return tweet

def evaluation_prediction(path_predictions, path_golds, metrics=['ICM', 'FMeasure'], verbose=False):
    test = PyEvALLEvaluation()
    params= dict()

    TASK_3_HIERARCHY = {"YES":["IDEOLOGICAL-INEQUALITY","STEREOTYPING- DOMINANCE","OBJECTIFICATION", "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL- VIOLENCE"], "NO":[]}
    params[PyEvALLUtils.PARAM_HIERARCHY]= TASK_3_HIERARCHY
    params[PyEvALLUtils.PARAM_REPORT]= PyEvALLUtils.PARAM_OPTION_REPORT_DATAFRAME  
    # or params[PyEvALLUtils.PARAM_REPORT]= PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED
    report= test.evaluate(path_predictions, path_golds, metrics, **params)
    
    avg_values = []
    classes_values = {}
    
    for metric in metrics:
        if metric == 'FMeasure': # for some reason pyevall change the names for the dataframes columns
            metric = 'F1'
        if metric == 'ICMSoft':
            metric = 'ICM-Soft'
        if metric == 'ICMNorm':
            metric = 'ICM-Norm'
        if metric == 'ICMSoftNorm':
            metric = 'ICM-Soft-Norm'
        
        avg_values.append(float(report.df_average[metric].iloc[0])) # retrieve the value of the metric from the dataframe

    if verbose:
        report.print_report()

    if not(report.df_test_case_classes is None) and (not report.df_test_case_classes.empty): # retrieve the value from the dataframe iff exist this dataframe
        classes_values = report.df_test_case_classes.drop(axis='columns', labels='files').to_dict('records')[0]
        
    return avg_values, classes_values

def compute_hardlabels(soft_pred:dict, soft_gold:dict):
    hard_pred = []
    hard_gold = []

    categories=['IDEOLOGICAL-INEQUALITY', 'STEREOTYPING-DOMINANCE',  
                'OBJECTIFICATION', 'SEXUAL-VIOLENCE', 'MISOGYNY-NON-SEXUAL-VIOLENCE']

    sexism_soft_gold = [soft_gold[key] for key in categories] # take all the soft values (same for gold)
    sexism_soft_pred = [soft_pred[key] for key in categories]

    if soft_pred['NO'] > 0.5: # apply the strange threshold of the challenge (also for gold)
        hard_pred.append('NO')
    elif soft_pred['NO'] == 0.5 and 0.5 in sexism_soft_pred:
        hard_pred = []
    else:
        hard_pred = [key for key in categories if soft_pred[key] > (1/5)]

    if soft_gold['NO'] > 0.5: 
        hard_gold.append('NO')
    elif soft_gold['NO'] == 0.5 and 0.5 in sexism_soft_gold:
        hard_gold = []
    else:
        hard_gold = [key for key in categories if soft_gold[key] > (1/5)]

    return hard_pred, hard_gold

def evaluate_model(model:nn.Module, data:DataLoader, hard_metrics=['ICM', 'FMeasure'], soft_metrics=['ICMSoft'], categories=['NO', 'IDEOLOGICAL-INEQUALITY', 'STEREOTYPING-DOMINANCE', 
                                                  'OBJECTIFICATION', 'SEXUAL-VIOLENCE', 'MISOGYNY-NON-SEXUAL-VIOLENCE']):
    
    # the followings are list of dict for the json challenge format (hard and soft mode)
    # in all the code gold means that it is for the gold labels
    hard_results = []
    soft_results = []
    hard_gold_results = []
    soft_gold_results = []
    id_json = 0

    model.eval()
    with torch.no_grad():
        for batch in data:
            input, labels = batch

            model_output = model(input)

            for output, label in zip(model_output, labels): # take all outputs and labels (one by one)
                soft_pred = {} # in the soft mode the key 'value' must have a dict of soft labels
                soft_gold = {}

                hard_pred = [] # in the hard mode the key 'value' must have a list of hard labels
                hard_gold = []

                for i, category in enumerate(categories):
                    soft_pred[category] = output[i].item() # retrieve the soft values 
                    soft_gold[category] = label[i].item()


                hard_pred, hard_gold = compute_hardlabels(soft_pred, soft_gold) # get the relative hard labels from the soft ones

                if hard_pred: # write hard dict iff there is at least one hard label (same for gold labels)
                    hard_results.append({"test_case": "EXIST2024", "id": str(id_json), "value": hard_pred})
                if hard_gold:
                    hard_gold_results.append({"test_case": "EXIST2024", "id": str(id_json), "value": hard_gold})

                soft_results.append({"test_case": "EXIST2024", "id": str(id_json), "value": soft_pred})
                soft_gold_results.append({"test_case": "EXIST2024", "id": str(id_json), "value": soft_gold})
                id_json += 1

    # write the list of dict in the relative tmp file, so PyEvall can use them to compute the metrics

    tmp_hard_gold = open("tmp_hard_gold.json", 'w')
    json.dump(hard_gold_results, tmp_hard_gold, indent=1) 
    tmp_hard_gold.close()

    tmp_hard = open("tmp_hard.json", 'w')
    json.dump(hard_results, tmp_hard, indent=1) 
    tmp_hard.close()

    tmp_soft_gold = open("tmp_soft_gold.json", 'w')
    json.dump(soft_gold_results, tmp_soft_gold, indent=1)
    tmp_soft_gold.close() 
    
    tmp_soft = open("tmp_soft.json", 'w')    
    json.dump(soft_results, tmp_soft, indent=1) 
    tmp_soft.close() 
    
    hard, soft = evaluation_prediction('tmp_hard.json', 'tmp_hard_gold.json', hard_metrics), evaluation_prediction('tmp_soft.json', 'tmp_soft_gold.json', soft_metrics)


    # just a piece of code to return 3 dict so it is easier to log with wandb
    hard_dict = {} 
    soft_dict = {}
    for i, metric in enumerate(hard_metrics):
        hard_dict[metric] = hard[0][i]

    for i, metric in enumerate(soft_metrics):
        soft_dict[metric] = soft[0][i]

    return hard_dict, hard[1], soft_dict 