from read_write_file import *

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

def extract_dataset(model_name = 'cardiffnlp/twitter-xlm-roberta-base-sentiment', \
                    input_file = ''):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    dataset = load_list_from_jsonl_file(input_file)

    label_list = ['negative', 'neutral', 'positive']

    for item in dataset:

        source = item['source']
        candidate = item['best_can']
        target = item['target']

        # source
        
        encoded_input = tokenizer(source, return_tensors='pt', max_length = 512)
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = list(softmax(scores))

        source_dict = {}
        for label, value in zip(label_list, scores):
            source_dict[label] = value  
        item['source_dict'] = source_dict

        # candidate
        encoded_input = tokenizer(candidate, return_tensors='pt', max_length = 512)
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = list(softmax(scores))

        candidate_dict = {}
        for label, value in zip(label_list, scores):
            candidate_dict[label] = value  
        item['candidate_dict'] = candidate_dict

        # target
        encoded_input = tokenizer(target, return_tensors='pt', max_length = 512)
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = list(softmax(scores))

        target_dict = {}
        for label, value in zip(label_list, scores):
            target_dict[label] = value  
        item['target_dict'] = target_dict

        print('item: ', item)
        print('-------------------------------')

    # save
    write_list_to_jsonl_file(input_file, dataset, file_access = 'w')
    

#....................
if __name__ == '__main__':
    '''extract_dataset(model_name = 'cardiffnlp/twitter-xlm-roberta-base-sentiment', \
                    input_file = 'dataset/phrase2/generated_test_para_256_diff_sim_best.json')'''
             
    '''extract_dataset(model_name = 'cardiffnlp/twitter-xlm-roberta-base-sentiment', \
                    input_file = 'dataset/phrase2/generated_validation_para_256_diff_sim_best.json')'''
    
    '''extract_dataset(model_name = 'cardiffnlp/twitter-xlm-roberta-base-sentiment', \
                    input_file = 'dataset/phrase2/generated_test_para_256_diff_rouge_best.json')'''             
    extract_dataset(model_name = 'cardiffnlp/twitter-xlm-roberta-base-sentiment', \
                    input_file = 'dataset/phrase2/generated_validation_para_256_diff_rouge_best.json')
    
    extract_dataset(model_name = 'cardiffnlp/twitter-xlm-roberta-base-sentiment', \
                    input_file = 'dataset/phrase2/generated_test_para_256_diff_simrouge_best.json')            
    extract_dataset(model_name = 'cardiffnlp/twitter-xlm-roberta-base-sentiment', \
                    input_file = 'dataset/phrase2/generated_validation_para_256_diff_simrouge_best.json')
                    
    extract_dataset(model_name = 'cardiffnlp/twitter-xlm-roberta-base-sentiment', \
                    input_file = 'dataset/phrase2/generated_test_para_256_random_sim_best.json')            
    extract_dataset(model_name = 'cardiffnlp/twitter-xlm-roberta-base-sentiment', \
                    input_file = 'dataset/phrase2/generated_validation_para_256_random_sim_best.json')
                    
    extract_dataset(model_name = 'cardiffnlp/twitter-xlm-roberta-base-sentiment', \
                    input_file = 'dataset/phrase2/generated_test_para_256_random_rouge_best.json')              
    extract_dataset(model_name = 'cardiffnlp/twitter-xlm-roberta-base-sentiment', \
                    input_file = 'dataset/phrase2/generated_validation_para_256_random_rouge_best.json')
                    
    extract_dataset(model_name = 'cardiffnlp/twitter-xlm-roberta-base-sentiment', \
                    input_file = 'dataset/phrase2/generated_test_para_256_random_simrouge_best.json')  
    extract_dataset(model_name = 'cardiffnlp/twitter-xlm-roberta-base-sentiment', \
                    input_file = 'dataset/phrase2/generated_validation_para_256_random_simrouge_best.json')
