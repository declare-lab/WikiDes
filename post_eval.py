import numpy as np
import sys
import argparse
import json
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

#from time import time
#from datetime import timedelta

import datasets
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer

from model import *
from data_utils import *

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device('cuda:0')

from read_write_file import *
from metrics import *
    
class Config:
    def __init__(self, model_name = 'bert-base-cased', tokenizer = 'bert-base-cased', batch_size = 16, \
                  max_length = 128):
        super(Config, self).__init__()

        self.seed = 42
        #self.model = BertModel.from_pretrained(model)
        #self.tokenizer = BertTokenizer.from_pretrained(model, do_lower_case = False)
        self.model_name = model_name
        self.tokenizer = tokenizer
    
        self.batch_size = batch_size
        self.max_length = max_length


def load_data(tokenizer, batch_size, max_length, input_file = ''):
    
    data = datasets.load_dataset('json', data_files = input_file)
    
    sources = []
    candidates = []
    targets = []
    for item in data['train']:
        sources.append(item['source'])
        candidates.append(item['candidate'])
        targets.append(item['target'])

    ds = PostEvalDataset(sources=np.array(sources),
                         candidates=np.array(candidates),
                         targets=np.array(targets),
                         tokenizer=tokenizer,
                         max_length=max_length)
    
    # collate_fn=lambda x: x
    return DataLoader(ds, batch_size=batch_size, num_workers=4)


def test_model_bertscore(model_name, split_type, model, tokenizer, batch_size = 64, max_length = 256, input_file = ''):

    #tokenizer = BertTokenizer.from_pretrained(config.tokenizer, do_lower_case = False)
    val_dataloader = load_data(tokenizer, batch_size, max_length, input_file)

    #checkpoint = torch.load('output/phrase2/model.bin')
    #model = checkpoint['model']
    #model.cuda()
    model.eval()

    count = 0
    predictions = []
    predictions_gold = []
    references = []

    with torch.no_grad():
        for (i, batch) in enumerate(val_dataloader):
       
            output = model(batch)
            can_score = output['candidate_score']
            target_score = output['target_score']
            max_ids = can_score.argmax(1)

            sources = batch['source']
            candidates = batch['candidate']
            targets = batch['target']

            for j in range(max_ids.size(-1)):
                count += 1
                source = sources[j]
                candidate = candidates[max_ids[j]][j]
                target = targets[j]

                other_candidates = [c[j] for c in candidates]
                
                print('--- source: ', source)
                print('--- candidate: ', candidate)
                print('--- other candidates: ', other_candidates)
                print('--- target: ', target)

                predictions.append(candidate)
                predictions_gold.append(target)
                references.append(source)

                print('--------------')

    bert_ouput = compute_bertscore_batch(predictions, references)
    bert_gold_ouput = compute_bertscore_batch(predictions_gold, references)
    
    result_dict = {}
    result_dict['file'] = input_file
    result_dict['split_type'] = split_type
    result_dict['model_name'] = model_name
    
    result_dict['bertscore_f1'] = bert_ouput['bertscore_f1']
    result_dict['bertscore_precision'] = bert_ouput['bertscore_precision']
    result_dict['bertscore_recall'] = bert_ouput['bertscore_recall']

    result_dict['bertscore_gold_f1'] = bert_gold_ouput['bertscore_f1']
    result_dict['bertscore_gold_precision'] = bert_gold_ouput['bertscore_precision']
    result_dict['bertscore_gold_recall'] = bert_gold_ouput['bertscore_recall']
    
    write_single_dict_to_json_file('output/phrase2/result.json', result_dict, file_access = 'a')
    
    return result_dict

def test_model(model_name, split_type, model, tokenizer, batch_size = 64, max_length = 256, input_file = ''):
    
    #tokenizer = BertTokenizer.from_pretrained(config.tokenizer, do_lower_case = False)
    val_dataloader = load_data(tokenizer, batch_size, max_length, input_file)

    #checkpoint = torch.load('output/phrase2/model.bin')
    #model = checkpoint['model']
    #model.cuda()
    model.eval()

    count = 0

    result_dict = {}
    
    
    result_dict['rouge1'] = 0
    result_dict['rouge2'] = 0
    result_dict['rougeL'] = 0
    
    result_dict['gold_rouge1'] = 0
    result_dict['gold_rouge2'] = 0
    result_dict['gold_rougeL'] = 0
    
    result_dict['meteor'] = 0
    result_dict['bleu'] = 0
    
    result_dict['meteor_gold'] = 0
    result_dict['bleu_gold'] = 0

    with torch.no_grad():
        for (i, batch) in enumerate(val_dataloader):
       
            output = model(batch)
            can_score = output['candidate_score']
            target_score = output['target_score']
            max_ids = can_score.argmax(1)

            sources = batch['source']
            candidates = batch['candidate']
            targets = batch['target']

            '''print('candidates: ', batch['candidate'])
            print('target: ', batch['target'])
            print('source: ', batch['source'])
            print('max_ids: ', max_ids)'''
            
            for j in range(max_ids.size(-1)):
                count += 1
                source = sources[j]
                candidate = candidates[max_ids[j]][j]
                target = targets[j]

                other_candidates = [c[j] for c in candidates]
                
                print('--- source: ', source)
                print('--- candidate: ', candidate)
                print('--- other candidates: ', other_candidates)
                print('--- target: ', target)

                rouge = compute_rouge_single(candidate, source)
                gold_rouge = compute_rouge_single(target, source)

                result_dict['rouge1'] += rouge['rouge1_fmeasure']
                result_dict['rouge2'] += rouge['rouge2_fmeasure']
                result_dict['rougeL'] += rouge['rougeL_fmeasure']

                result_dict['gold_rouge1'] += gold_rouge['rouge1_fmeasure']
                result_dict['gold_rouge2'] += gold_rouge['rouge2_fmeasure']
                result_dict['gold_rougeL'] += gold_rouge['rougeL_fmeasure']
                
                output = compute_bleu_single(candidate, source)
                result_dict['bleu'] += output['bleu']
                
                print('output_bleu: ', output['bleu'])

                output = compute_meteor_single(candidate, source)
                result_dict['meteor'] += output['meteor']
                
                print('output_metor: ', output['meteor'])
                
                output = compute_bleu_single(target, source)
                result_dict['bleu_gold'] += output['bleu']

                output = compute_meteor_single(target, source)
                result_dict['meteor_gold'] += output['meteor']

                print('rouge: ', rouge)
                print('--------------')

        
            #print('----------------------------------------')

    result_dict['rouge1'] = result_dict['rouge1'] / count
    result_dict['rouge2'] = result_dict['rouge2'] / count
    result_dict['rougeL'] = result_dict['rougeL'] / count

    result_dict['gold_rouge1'] = result_dict['gold_rouge1'] / count
    result_dict['gold_rouge2'] = result_dict['gold_rouge2'] / count
    result_dict['gold_rougeL'] = result_dict['gold_rougeL'] / count
    
    result_dict['bleu'] = result_dict['bleu'] / count
    result_dict['meteor'] = result_dict['meteor'] / count
    
    result_dict['bleu_gold'] = result_dict['bleu_gold'] / count
    result_dict['meteor_gold'] = result_dict['meteor_gold'] / count

    metrics = [v for k, v in result_dict.items()]
    loss = 1 - sum(metrics)/len(metrics)

    result_dict['model_name'] = model_name
    result_dict['input_file'] = input_file
    result_dict['split_type'] = split_type
    result_dict['loss'] = loss
    
    print('result_dict: ', result_dict)
    print('loss: ', loss)
    
    # write file
    
    write_single_dict_to_json_file('output/phrase2/result.json', result_dict, file_access = 'a')

    #return loss, result_dict

def eval_model(model, tokenizer, batch_size = 64, max_length = 256, input_file = ''):
    
    #tokenizer = BertTokenizer.from_pretrained(config.tokenizer, do_lower_case = False)
    val_dataloader = load_data(tokenizer, batch_size, max_length, input_file)

    #checkpoint = torch.load('output/phrase2/model.bin')
    #model = checkpoint['model']
    #model.cuda()
    model.eval()

    count = 0

    result_dict = {}
    result_dict['rouge1'] = 0
    result_dict['rouge2'] = 0
    result_dict['rougeL'] = 0
    
    result_dict['gold_rouge1'] = 0
    result_dict['gold_rouge2'] = 0
    result_dict['gold_rougeL'] = 0
    
    with torch.no_grad():
        for (i, batch) in enumerate(val_dataloader):

            output = model(batch)
            can_score = output['candidate_score']
            target_score = output['target_score']
            max_ids = can_score.argmax(1)

            sources = batch['source']
            candidates = batch['candidate']
            targets  = batch['target']

            '''print('candidates: ', batch['candidate'])
            print('target: ', batch['target'])
            print('source: ', batch['source'])
            print('max_ids: ', max_ids)'''
            
            for j in range(max_ids.size(-1)):
                count += 1
                source = sources[j]
                candidate = candidates[max_ids[j]][j]
                target = targets[j]

                other_candidates = [c[j] for c in candidates]
                
                print('--- source: ', source)
                print('--- candidate: ', candidate)
                print('--- other candidates: ', other_candidates)
                print('--- target: ', target)

                rouge = compute_rouge_single(candidate, source)
                gold_rouge = compute_rouge_single(target, source)

                result_dict['rouge1'] += rouge['rouge1_fmeasure']
                result_dict['rouge2'] += rouge['rouge2_fmeasure']
                result_dict['rougeL'] += rouge['rougeL_fmeasure']

                result_dict['gold_rouge1'] += gold_rouge['rouge1_fmeasure']
                result_dict['gold_rouge2'] += gold_rouge['rouge2_fmeasure']
                result_dict['gold_rougeL'] += gold_rouge['rougeL_fmeasure']

                print('rouge: ', rouge)
                print('--------------')

        
            #print('----------------------------------------')

    result_dict['rouge1'] = result_dict['rouge1'] / count
    result_dict['rouge2'] = result_dict['rouge2'] / count
    result_dict['rougeL'] = result_dict['rougeL'] / count

    result_dict['gold_rouge1'] = result_dict['gold_rouge1'] / count
    result_dict['gold_rouge2'] = result_dict['gold_rouge2'] / count
    result_dict['gold_rougeL'] = result_dict['gold_rougeL'] / count

    metrics = [v for k, v in result_dict.items()]
    loss = 1 - sum(metrics)/len(metrics)
    print('result_dict: ', result_dict)
    print('eval_loss: ', loss)
    model.train()

    return loss, result_dict
 

def select_best_can_model(model, tokenizer, batch_size = 64, max_length = 256, input_file = '', output_file = ''):    
    
    #tokenizer = BertTokenizer.from_pretrained(config.tokenizer, do_lower_case = False)
    val_dataloader = load_data(tokenizer, batch_size, max_length, input_file)

    #checkpoint = torch.load('output/phrase2/model.bin')
    #model = checkpoint['model']
    #model.cuda()
    model.eval()

    data_list = []
    
    with torch.no_grad():
        for (i, batch) in enumerate(val_dataloader):
            
            print('i: ', i)
            output = model(batch)
            can_score = output['candidate_score']
            target_score = output['target_score']
            max_ids = can_score.argmax(1)

            sources = batch['source']
            candidates = batch['candidate']
            targets  = batch['target']
            
            for j in range(max_ids.size(-1)):
                
                source = sources[j]
                candidate = candidates[max_ids[j]][j]
                target = targets[j]

                other_candidates = [c[j] for c in candidates]
                
                data_list.append({'source':source, 'best_can':candidate, 'target':target})
            #print('--------------')

    # write file
    write_list_to_jsonl_file(output_file, data_list, file_access = 'w')
    
    
 
def train_model(config, splitting_type = 'different', num_epochs = 5, use_sim = True, use_rouge = True, training_file = '', val_file = '', test_file = ''):

    if ('roberta' in config.tokenizer):
        tokenizer = RobertaTokenizer.from_pretrained(config.tokenizer, do_lower_case = False)
        
    else:
        tokenizer = BertTokenizer.from_pretrained(config.tokenizer, do_lower_case = False)
    
    train_dataloader = load_data(tokenizer, config.batch_size, config.max_length, input_file = training_file)
  
    # configure model
    model = PostEvalModel(config.model_name, use_sim = use_sim, use_rouge = use_rouge, hidden_size=768)
    model.cuda()
    model.train()

    # lr=0 to rank the candadiates since we only use embeddings similarity
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0)

    best_loss = 1
    history = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        total_loss = 0
        count = 0
        for (i, batch) in enumerate(train_dataloader):

            #print('batch: ', batch)
            output = model(batch)
            can_score = output['candidate_score']
            target_score = output['target_score']
            loss = RankingLoss(can_score, target_score, use_can_loss = True, can_margin = 0.01, gold_margin = 0.01)
            total_loss += loss.item()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            # free memory
            del can_score, target_score, loss

      
        # evaluation
        print('total_loss: ', total_loss)
        eval_loss, eval_metric = eval_model(model, tokenizer, input_file = val_file)
        test_loss, test_metric = eval_model(model, tokenizer, input_file = test_file)
        
        print('----------------------------------------')

        # save history file
        history_item = {
                'epoch': epoch + 1,
                'train_loss': total_loss,
                'eval_loss': eval_loss,
                'eval_metric': eval_metric,
                'test_loss': test_loss,
                'test_metric': test_metric
                }
        history.append(history_item)
        # metric string
        metric_str = ''
        if (use_sim == True): metric_str += 'sim'
        if (use_rouge == True): metric_str += 'rouge'
        
        write_single_dict_to_json_file('output/phrase2/history_' + config.model_name + '_' + metric_str + '_' + splitting_type +  '.json', history_item)

        # save model with best eval_loss
        if (eval_loss < best_loss): # the least the better
            best_loss = eval_loss

            # save model
            from pathlib import Path
            Path("output/phrase2").mkdir(parents=True, exist_ok=True)
            torch.save({'model_name': config.model_name,
                        'history': history,
                        'model': model}, 'output/phrase2/' + config.model_name + '_' + metric_str + '_' +  splitting_type  + '.bin')

        
if __name__ == '__main__':

    '''config = Config(model_name = 'bert-base-cased', tokenizer = 'bert-base-cased', batch_size = 2, max_length = 256)
    train_model(config, splitting_type = 'different', num_epochs = 3, use_sim = True, use_rouge = False, \
                training_file = 'dataset/phrase2/generated_training_para_256_diff.json', \
                val_file = 'dataset/phrase2/generated_validation_para_256_diff.json', \
                test_file = 'dataset/phrase2/generated_test_para_256_diff.json')'''
    
    '''config = Config(model_name = 'bert-base-cased', tokenizer = 'bert-base-cased', batch_size = 2, max_length = 256)
    train_model(config, splitting_type = 'different', num_epochs = 3, use_sim = False, use_rouge = True, \
                training_file = 'dataset/phrase2/generated_training_para_256_diff.json', \
                val_file = 'dataset/phrase2/generated_validation_para_256_diff.json', \
                test_file = 'dataset/phrase2/generated_test_para_256_diff.json')'''

    '''config = Config(model_name = 'bert-base-cased', tokenizer = 'bert-base-cased', batch_size = 2, max_length = 256)
    train_model(config, splitting_type = 'different', num_epochs = 1, use_sim = True, use_rouge = True, \
                training_file = 'dataset/phrase2/generated_training_para_256_diff.json', \
                val_file = 'dataset/phrase2/generated_validation_para_256_diff.json', \
                test_file = 'dataset/phrase2/generated_test_para_256_diff.json')'''
    
    '''config = Config(model_name = 'bert-base-cased', tokenizer = 'bert-base-cased', batch_size = 2, max_length = 256)
    train_model(config, splitting_type = 'random', num_epochs = 3, use_sim = True, use_rouge = False, \
                training_file = 'dataset/phrase2/generated_training_para_256_random.json', \
                val_file = 'dataset/phrase2/generated_validation_para_256_random.json', \
                test_file = 'dataset/phrase2/generated_test_para_256_random.json')
    
    config = Config(model_name = 'bert-base-cased', tokenizer = 'bert-base-cased', batch_size = 2, max_length = 256)
    train_model(config, splitting_type = 'random', num_epochs = 1, use_sim = False, use_rouge = True, \
                training_file = 'dataset/phrase2/generated_training_para_256_random.json', \
                val_file = 'dataset/phrase2/generated_validation_para_256_random.json', \
                test_file = 'dataset/phrase2/generated_test_para_256_random.json')'''
    
    '''config = Config(model_name = 'bert-base-cased', tokenizer = 'bert-base-cased', batch_size = 2, max_length = 256)
    train_model(config, splitting_type = 'random', num_epochs = 1, use_sim = True, use_rouge = True, \
                training_file = 'dataset/phrase2/generated_training_para_256_random.json', \
                val_file = 'dataset/phrase2/generated_validation_para_256_random.json', \
                test_file = 'dataset/phrase2/generated_test_para_256_random.json')'''


    # test cases
    # bert-base-cased_sim_different --------
    config = Config(model_name = 'bert-base-cased', tokenizer = 'bert-base-cased', batch_size = 4, max_length = 256)
    tokenizer = BertTokenizer.from_pretrained(config.tokenizer, do_lower_case = False)
    
    checkpoint = torch.load('output/phrase2/bert-base-cased_sim_different.bin')
    model = checkpoint['model']
    model.cuda()
    model.eval()
    
    select_best_can_model(model, tokenizer, batch_size = config.batch_size, max_length = config.max_length, \
                         input_file = 'dataset/phrase2/generated_test_para_256_diff.json', \
                          output_file = 'dataset/phrase2/generated_test_para_256_diff_sim_best.json')

    select_best_can_model(model, tokenizer, batch_size = config.batch_size, max_length = config.max_length, \
                         input_file = 'dataset/phrase2/generated_validation_para_256_diff.json', \
                          output_file = 'dataset/phrase2/generated_validation_para_256_diff_sim_best.json')

    '''test_model_bertscore(config.model_name, 'sim_different', model, tokenizer, batch_size = config.batch_size, \
               max_length = config.max_length, input_file = 'dataset/phrase2/generated_test_para_256_diff.json')
    test_model_bertscore(config.model_name, 'sim_different', model, tokenizer, batch_size = config.batch_size, \
               max_length = config.max_length, input_file = 'dataset/phrase2/generated_validation_para_256_diff.json')
   
    test_model(config.model_name, 'sim_different', model, tokenizer, batch_size = config.batch_size, \
               max_length = config.max_length, input_file = 'dataset/phrase2/generated_test_para_256_diff.json')
    test_model(config.model_name, 'sim_different', model, tokenizer, batch_size = config.batch_size, \
               max_length = config.max_length, input_file = 'dataset/phrase2/generated_validation_para_256_diff.json')'''
    
    del model, tokenizer, config 
    
    # ---------------------------------------
    
    # bert-base-cased_rouge_different -------     
    config = Config(model_name = 'bert-base-cased', tokenizer = 'bert-base-cased', batch_size = 4, max_length = 256)
    tokenizer = BertTokenizer.from_pretrained(config.tokenizer, do_lower_case = False)
    
    checkpoint = torch.load('output/phrase2/bert-base-cased_rouge_different.bin')
    model = checkpoint['model']
    model.cuda()
    model.eval()

    select_best_can_model(model, tokenizer, batch_size = config.batch_size, max_length = config.max_length, \
                         input_file = 'dataset/phrase2/generated_test_para_256_diff.json', \
                          output_file = 'dataset/phrase2/generated_test_para_256_diff_rouge_best.json')

    select_best_can_model(model, tokenizer, batch_size = config.batch_size, max_length = config.max_length, \
                         input_file = 'dataset/phrase2/generated_validation_para_256_diff.json', \
                          output_file = 'dataset/phrase2/generated_validation_para_256_diff_rouge_best.json')

    '''test_model_bertscore(config.model_name, 'rouge_different', model, tokenizer, batch_size = config.batch_size, \
               max_length = config.max_length, input_file = 'dataset/phrase2/generated_test_para_256_diff.json')
    test_model_bertscore(config.model_name, 'rouge_different', model, tokenizer, batch_size = config.batch_size, \
               max_length = config.max_length, input_file = 'dataset/phrase2/generated_validation_para_256_diff.json')
    
    test_model(config.model_name, 'rouge_different', model, tokenizer, batch_size = config.batch_size, \
               max_length = config.max_length, input_file = 'dataset/phrase2/generated_test_para_256_diff.json')
    test_model(config.model_name, 'rouge_different', model, tokenizer, batch_size = config.batch_size, \
               max_length = config.max_length, input_file = 'dataset/phrase2/generated_validation_para_256_diff.json')'''
    
    del model, tokenizer, config 
    # ---------------------------------------
    
    # bert-base-cased_simrouge_different -------     
    config = Config(model_name = 'bert-base-cased', tokenizer = 'bert-base-cased', batch_size = 4, max_length = 256)
    tokenizer = BertTokenizer.from_pretrained(config.tokenizer, do_lower_case = False)
    
    checkpoint = torch.load('output/phrase2/bert-base-cased_simrouge_different.bin')
    model = checkpoint['model']
    model.cuda()
    model.eval()

    select_best_can_model(model, tokenizer, batch_size = config.batch_size, max_length = config.max_length, \
                         input_file = 'dataset/phrase2/generated_test_para_256_diff.json', \
                          output_file = 'dataset/phrase2/generated_test_para_256_diff_simrouge_best.json')

    select_best_can_model(model, tokenizer, batch_size = config.batch_size, max_length = config.max_length, \
                         input_file = 'dataset/phrase2/generated_validation_para_256_diff.json', \
                          output_file = 'dataset/phrase2/generated_validation_para_256_diff_simrouge_best.json')

    '''test_model_bertscore(config.model_name, 'simrouge_different', model, tokenizer, batch_size = config.batch_size, \
               max_length = config.max_length, input_file = 'dataset/phrase2/generated_test_para_256_diff.json')
    test_model_bertscore(config.model_name, 'simrouge_different', model, tokenizer, batch_size = config.batch_size, \
               max_length = config.max_length, input_file = 'dataset/phrase2/generated_validation_para_256_diff.json')
    
    test_model(config.model_name, 'simrouge_different', model, tokenizer, batch_size = config.batch_size, \
               max_length = config.max_length, input_file = 'dataset/phrase2/generated_test_para_256_diff.json')
    test_model(config.model_name, 'simrouge_different', model, tokenizer, batch_size = config.batch_size, \
               max_length = config.max_length, input_file = 'dataset/phrase2/generated_validation_para_256_diff.json')'''
               
    del model, tokenizer, config    
    # ---------------------------------------
    
    # bert-base-cased_sim_random -------     
    config = Config(model_name = 'bert-base-cased', tokenizer = 'bert-base-cased', batch_size = 4, max_length = 256)
    tokenizer = BertTokenizer.from_pretrained(config.tokenizer, do_lower_case = False)
    
    checkpoint = torch.load('output/phrase2/bert-base-cased_sim_random.bin')
    model = checkpoint['model']
    model.cuda()
    model.eval()
    

    select_best_can_model(model, tokenizer, batch_size = config.batch_size, max_length = config.max_length, \
                         input_file = 'dataset/phrase2/generated_test_para_256_random.json', \
                          output_file = 'dataset/phrase2/generated_test_para_256_random_sim_best.json')

    select_best_can_model(model, tokenizer, batch_size = config.batch_size, max_length = config.max_length, \
                         input_file = 'dataset/phrase2/generated_validation_para_256_random.json', \
                          output_file = 'dataset/phrase2/generated_validation_para_256_random_sim_best.json')

    '''test_model_bertscore(config.model_name, 'sim_random', model, tokenizer, batch_size = config.batch_size, \
               max_length = config.max_length, input_file = 'dataset/phrase2/generated_test_para_256_random.json')
    test_model_bertscore(config.model_name, 'sim_random', model, tokenizer, batch_size = config.batch_size, \
               max_length = config.max_length, input_file = 'dataset/phrase2/generated_validation_para_256_random.json')
    
    test_model(config.model_name, 'sim_random', model, tokenizer, batch_size = config.batch_size, \
               max_length = config.max_length, input_file = 'dataset/phrase2/generated_test_para_256_random.json')
    test_model(config.model_name, 'sim_random', model, tokenizer, batch_size = config.batch_size, \
               max_length = config.max_length, input_file = 'dataset/phrase2/generated_validation_para_256_random.json')'''
    
    del model, tokenizer, config
    # ---------------------------------------
    
    # bert-base-cased_rouge_random -------     
    config = Config(model_name = 'bert-base-cased', tokenizer = 'bert-base-cased', batch_size = 4, max_length = 256)
    tokenizer = BertTokenizer.from_pretrained(config.tokenizer, do_lower_case = False)
    
    checkpoint = torch.load('output/phrase2/bert-base-cased_rouge_random.bin')
    model = checkpoint['model']
    model.cuda()
    model.eval()

    select_best_can_model(model, tokenizer, batch_size = config.batch_size, max_length = config.max_length, \
                         input_file = 'dataset/phrase2/generated_test_para_256_random.json', \
                          output_file = 'dataset/phrase2/generated_test_para_256_random_rouge_best.json')

    select_best_can_model(model, tokenizer, batch_size = config.batch_size, max_length = config.max_length, \
                         input_file = 'dataset/phrase2/generated_validation_para_256_random.json', \
                          output_file = 'dataset/phrase2/generated_validation_para_256_random_rouge_best.json')

    '''test_model_bertscore(config.model_name, 'rouge_random', model, tokenizer, batch_size = config.batch_size, \
               max_length = config.max_length, input_file = 'dataset/phrase2/generated_test_para_256_random.json')
    test_model_bertscore(config.model_name, 'rouge_random', model, tokenizer, batch_size = config.batch_size, \
               max_length = config.max_length, input_file = 'dataset/phrase2/generated_validation_para_256_random.json')
    
    test_model(config.model_name, 'rouge_random', model, tokenizer, batch_size = config.batch_size, \
               max_length = config.max_length, input_file = 'dataset/phrase2/generated_test_para_256_random.json')
    test_model(config.model_name, 'rouge_random', model, tokenizer, batch_size = config.batch_size, \
               max_length = config.max_length, input_file = 'dataset/phrase2/generated_validation_para_256_random.json')'''
    
    del model, tokenizer, config 
    # ---------------------------------------
    
    # bert-base-cased_simrouge_random -------     
    config = Config(model_name = 'bert-base-cased', tokenizer = 'bert-base-cased', batch_size = 4, max_length = 256)
    tokenizer = BertTokenizer.from_pretrained(config.tokenizer, do_lower_case = False)
    
    checkpoint = torch.load('output/phrase2/bert-base-cased_simrouge_random.bin')
    model = checkpoint['model']
    model.cuda()
    model.eval()

    select_best_can_model(model, tokenizer, batch_size = config.batch_size, max_length = config.max_length, \
                         input_file = 'dataset/phrase2/generated_test_para_256_random.json', \
                          output_file = 'dataset/phrase2/generated_test_para_256_random_simrouge_best.json')

    select_best_can_model(model, tokenizer, batch_size = config.batch_size, max_length = config.max_length, \
                         input_file = 'dataset/phrase2/generated_validation_para_256_random.json', \
                          output_file = 'dataset/phrase2/generated_validation_para_256_random_simrouge_best.json')

    '''test_model_bertscore(config.model_name, 'simrouge_random', model, tokenizer, batch_size = config.batch_size, \
               max_length = config.max_length, input_file = 'dataset/phrase2/generated_test_para_256_random.json')
    test_model_bertscore(config.model_name, 'simrouge_random', model, tokenizer, batch_size = config.batch_size, \
               max_length = config.max_length, input_file = 'dataset/phrase2/generated_validation_para_256_random.json')
    
    test_model(config.model_name, 'simrouge_random', model, tokenizer, batch_size = config.batch_size, \
               max_length = config.max_length, input_file = 'dataset/phrase2/generated_test_para_256_random.json')
    test_model(config.model_name, 'simrouge_random', model, tokenizer, batch_size = config.batch_size, \
               max_length = config.max_length, input_file = 'dataset/phrase2/generated_validation_para_256_random.json')'''
    
    del model, tokenizer, config
    # ---------------------------------------
