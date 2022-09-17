from transformers import AutoTokenizer

from metrics import *
from read_write_file import *

import numpy as np
import matplotlib.pyplot as plt

import string
PUNCTS = string.punctuation + '”“¿?.�✔✅⤵➕➖⛔✍⃣.-'
PUNCTS = list(PUNCTS)

import spacy
nlp_en = spacy.load('en_core_web_md')
STOPWORDS = nlp_en.Defaults.stop_words


'''def check_dataset(input_file = 'dataset/phrase1/training_para_256.json'):

    dataset = load_list_from_jsonl_file(input_file)
    vocab_list = []

    for item in dataset:
        des = item['target']
        para = item['source']
        instances = item['baseline_candidates']
        if (len(instances) == 0):
            print(item)
            print('----------------')
    
    return vocab_list'''

def count_vocab_size(input_file = 'dataset/collected_data.json'):

    dataset = load_list_from_json_file(input_file, format_json = False)
    vocab_list = []

    for item in dataset:
        des = item['description']
        para = item['first_paragraph']
        doc = nlp_en(para + ' ' + des)
        
        for token in doc:
            if (token.text.strip() not in vocab_list and token.text.strip() != '' and token.text.strip() not in PUNCTS):
                vocab_list.append(token.text.strip())

    print('vocab_list: ', len(vocab_list))
    #write_list_to_json_file('dataset/instance_distribution.json', instance_dict, file_access = 'w')
    
    return vocab_list   


def text_by_len(input_file = 'dataset/collected_data.json'):

    dataset = load_list_from_json_file(input_file, format_json = False)
    target_dict = {}
    source_dict = {}  
    
    for item in dataset:

        # count words in descriptions and paragraphs
        doc_des = nlp_en(item['description'])
        doc_des = [token.text.strip() for token in doc_des if token.text.strip() not in PUNCTS and token.text.strip() != '']

        if (len(doc_des) not in target_dict): target_dict[len(doc_des)] = 1
        else: target_dict[len(doc_des)] += 1

        doc_para = nlp_en(item['first_paragraph'])
        doc_para = [token.text.strip() for token in doc_para if token.text.strip() not in PUNCTS and token.text.strip() != '']

        if (len(doc_para) not in source_dict): source_dict[len(doc_para)] = 1
        else: source_dict[len(doc_para)] += 1

        '''print('doc_des: ', doc_des)
        print('doc_para: ', doc_para)
        print('----------------------------')'''

    source_dict = dict(sorted(source_dict.items(), key = lambda x:x[0], reverse = True))
    target_dict = dict(sorted(target_dict.items(), key = lambda x:x[0], reverse = True))
    
    avg_source_length = sum([item[0]*item[1] for item in source_dict.items()])/len(dataset)
    avg_target_length = sum([item[0]*item[1] for item in target_dict.items()])/len(dataset)

    print('source_dict: ', len(source_dict), source_dict)
    print('target_dict: ', len(target_dict), target_dict)

    print('avg_source_length: ', avg_source_length)
    print('avg_target_length: ', avg_target_length)
    
    #show_token_len_plot(source_dict, target_dict, x_label = 'Text length', y_label = 'Number of texts')
    
    return source_dict, target_dict


def text_by_token(input_file = 'dataset/collected_data.json'):
    """
        use "bert-base-cased" to count tokens
    """

    dataset = load_list_from_json_file(input_file, format_json = False)
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base', do_lower_case=False)   
    source_dict = {}
    target_dict = {}

    for item in dataset:

        hyp_list = tokenizer.tokenize(item['description'])
        if (len(hyp_list) not in source_dict): source_dict[len(hyp_list)] = 1
        else: source_dict[len(hyp_list)] += 1

        ref_list = tokenizer.tokenize(item['first_paragraph'])
        if (len(ref_list) not in target_dict): target_dict[len(ref_list)] = 1
        else: target_dict[len(ref_list)] += 1
        
    # sort dict
    source_dict = dict(sorted(source_dict.items(), key = lambda x:x[0], reverse = True))
    target_dict = dict(sorted(target_dict.items(), key = lambda x:x[0], reverse = True))
    
    avg_source_length = sum([item[0]*item[1] for item in source_dict.items()])/len(dataset)
    avg_target_length = sum([item[0]*item[1] for item in target_dict.items()])/len(dataset)

    print('source_dict: ', len(source_dict), source_dict)
    print('target_dict: ', len(target_dict), target_dict)

    print('avg_source_length: ', avg_source_length)
    print('avg_target_length: ', avg_target_length)
    show_token_len_plot(source_dict, target_dict, 'Number of tokens', 'Number of texts')

    return source_dict, target_dict, avg_source_length, avg_target_length


def token_pos_by_para(input_file = 'dataset/collected_data.json'):

    dataset = load_list_from_json_file(input_file, format_json = False)
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base', do_lower_case=False)   

    pos_dict = {}
    for item in dataset:

        hyp_list = tokenizer.tokenize(item['description'])
        hyp_list = [token.replace('Ġ', '') for token in hyp_list] # remove Ġ
        hyp_list = [token for token in hyp_list if token not in STOPWORDS and token not in PUNCTS]

        ref_list = tokenizer.tokenize(item['first_paragraph'])
        ref_list = [token.replace('Ġ', '') for token in ref_list] # remove Ġ

        #print('hyp_list: ', hyp_list)
        #print('ref_list: ', ref_list)
        
        for token in hyp_list:

            indices = [i for i, x in enumerate(ref_list) if x == token]
            for i in indices:
                if (i not in pos_dict): pos_dict[i] = 1
                else: pos_dict[i] += 1     
        #print('---------------------------')

    pos_dict = dict(sorted(pos_dict.items(), key = lambda x:x[0]))
    print('pos_dict: ', len(pos_dict), pos_dict)

    from sklearn.preprocessing import MinMaxScaler

    pos_list = [[k] for k, v in pos_dict.items()]
    pos_list = MinMaxScaler(feature_range = (0,100)).fit(pos_list).transform(pos_list)
    pos_list = [pos[0] for pos in pos_list]
    val_list = [v for k, v in pos_dict.items()]

    # create dict
    rel_dict = {}
    for i in range(1,101):
        if (i%10 == 0):
            rel_dict[i] = 0

    # group by keys
    for p, v in zip(pos_list, val_list):
        for k, _ in rel_dict.items():
            if (p <= k and p > k - 10): rel_dict[k] += v

    # normalize frequency
    freq_list = [v for k, v in rel_dict.items()]
    real_freq_list = freq_list[:]
    
    total_freq = sum(freq_list)
    freq_list.append(total_freq)
    freq_list = [[v] for v in freq_list]
    freq_list = MinMaxScaler(feature_range = (0,100)).fit(freq_list).transform(freq_list)
    freq_list = [v[0] for v in freq_list[0:-1]] # remove the normalized total_freq

    for item, freq in zip(list(rel_dict.items()), freq_list):
        rel_dict[item[0]] = freq
    
    '''print('pos_list: ', pos_list)
    print('val_list: ', val_list)
    print('freq_list: ', freq_list)
    print('rel_dict: ', rel_dict)'''

    print('rel_dict: ', rel_dict)
    print('real_freq_list: ', real_freq_list)
    show_token_pos_plot(rel_dict, real_freq_list)

def show_token_pos_plot(pos_dict, real_values, x_label = 'Relative position (%)', y_label = 'Ratio (%)'):

    label_list, value_list = [], []
    for k, v in pos_dict.items():
        label_list.append(k)
        value_list.append(v)

    print('value_list: ', value_list, len(value_list))
    print('label_list: ', label_list, len(label_list))

    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(12,5))
    
    ax.bar(label_list, value_list, width=4)

    print('ax.containers[0]: ', ax.containers[0], type(ax.containers[0]))

    # format value_list
    for i, v in enumerate(value_list):
        vr = round(v, 2)
        if (vr < 0.1): value_list[i] = '< 0.1%'
        else: value_list[i] = str(vr) + '%'

    print('value_list: ', value_list)
    
    ax.bar_label(ax.containers[0], labels = value_list, label_type='edge')
    ax.set_xticks(label_list)
    label_list = [str(l) for l in label_list]
    ax.set_xticklabels(label_list)
    

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.grid(color='gray', linestyle='--', linewidth=0.7, axis='y')
    plt.xlim([0, 105])
    
    #plt.legend()
    plt.show()  

def show_token_len_plot(source_dict, target_dict, x_label = 'Text length', y_label = 'Number of texts'):

    # 661: 1, len 661 appear 1 time
    
    label_list, value_list = [], []
    for k, v in source_dict.items():
        label_list.append(k)
        value_list.append(v)

    label_list2, value_list2 = [], []
    print('value_list2: ', value_list2)
    for k, v in target_dict.items():
        label_list2.append(k)
        value_list2.append(v)

    print('len2: ', len(label_list2))
    print('len1: ', len(label_list))

    for i in range(len(label_list2), len(label_list)):
        print(i)
        label_list2.append(i)
        value_list2.append(0)

    print('value_list2: ', value_list2, len(value_list2))
    print('value_list: ', value_list, len(value_list))

    print('len2: ', len(label_list2))
    print('len1: ', len(label_list))
    
    label_list = np.array(label_list)
    label_list2 = np.array(label_list2)


    plt.rcParams.update({'font.size': 12})
    
    plt.bar(label_list - 0.2, value_list, label ='Paragraphs')
    plt.bar(label_list2 + 0.2, value_list2, label = 'Descriptions' )

    
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    '''for i in range(len(label_list)):
        plt.annotate(str(value_list[i]) + ' (' + str(round(percent_list[i]*100, 2)) + ' %)',
                     xy=(label_list[i],value_list[i]), ha='center', va='bottom')'''
    #plt.grid()
    plt.xlim([0, 500])
    #plt.figure(figsize=(8,5))
    plt.legend()
    plt.show()



def compute_dataset(input_file = 'dataset/phrase1/training_para_128.json', max_len = 256):
    dataset = load_list_from_jsonl_file(input_file)
    print(len(dataset))
    
    r1_pre, r2_pre, rL_pre = [], [], []
    r1_re, r2_re, rL_re = [], [], []
    r1_fm, r2_fm, rL_fm = [], [], []

    #bert_pre, bert_re, bert_f1 = [], [], []


    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base', do_lower_case=False)    
    for item in dataset[0:5000]: # get 5000 items only

        hyp_list = tokenizer.tokenize(item['target'])

        
        hyp_list = [token.replace('Ġ', '') for token in hyp_list] # remove Ġ
        #hyp_list = [token for token in hyp_list if token not in STOPWORDS and '##' not in token]
        hyp_list = [token for token in hyp_list if token not in STOPWORDS and token not in PUNCTS]
        hyp_text = ' '.join(w for w in hyp_list)
        print('hyp_text: ', hyp_text)


        ref_list = tokenizer.tokenize(item['source'])[:max_len] # limit length
        ref_list = [token.replace('Ġ', '') for token in ref_list] # remove Ġ
        ref_list = [token for token in ref_list if token not in STOPWORDS and token not in PUNCTS]
        ref_text = ' '.join(w for w in ref_list)
        print('ref_text: ', ref_text)
        
        output = compute_rouge_single(hyp_text, ref_text)

        r1_pre.append(output['rouge1_precision'])
        r2_pre.append(output['rouge2_precision'])
        rL_pre.append(output['rougeL_precision'])

        r1_re.append(output['rouge1_recall'])
        r2_re.append(output['rouge2_recall'])
        rL_re.append(output['rougeL_recall'])

        r1_fm.append(output['rouge1_fmeasure'])
        r2_fm.append(output['rouge2_fmeasure'])
        rL_fm.append(output['rougeL_fmeasure'])

        print('item: ', item)
        print('-----------------------------')

    '''sources, targets = [], []
    for item in dataset:
        sources.append(item['source'])
        targets.append(item['target'])

    output = compute_bertscore_batch(sources, targets)
    bert_pre = output['bertscore_precision']
    bert_re = output['bertscore_recall']
    bert_f1 = output['bertscore_f1']'''
        
    r1_pre = sum(r1_pre)/len(r1_pre)
    r2_pre = sum(r2_pre)/len(r2_pre)
    rL_pre = sum(rL_pre)/len(rL_pre)
    
    r1_re = sum(r1_re)/len(r1_re)
    r2_re = sum(r2_re)/len(r2_re)
    rL_re = sum(rL_re)/len(rL_re)
    
    r1_fm = sum(r1_fm)/len(r1_fm)
    r2_fm = sum(r2_fm)/len(r2_fm)
    rL_fm = sum(rL_fm)/len(rL_fm)

    '''bert_pre = sum(bert_pre)/len(bert_pre)
    bert_re = sum(bert_re)/len(bert_re)
    bert_f1 = sum(bert_f1)/len(bert_f1)'''

    result_dict = {}
    result_dict['max_len'] = max_len
    result_dict['avg_rouge1_precision'] = r1_pre
    result_dict['avg_rouge2_precision'] = r2_pre
    result_dict['avg_rougeL_precision'] = rL_pre

    result_dict['avg_rouge1_recall'] = r1_re
    result_dict['avg_rouge2_recall'] = r2_re
    result_dict['avg_rougeL_recall'] = rL_re

    result_dict['avg_rouge1_fm'] = r1_fm
    result_dict['avg_rouge2_fm'] = r2_fm
    result_dict['avg_rougeL_fm'] = rL_fm

    '''result_dict['avg_bertscore_precision'] = bert_pre
    result_dict['avg_bertscore_recall'] = bert_re
    result_dict['avg_bertscore_f1'] = bert_f1'''

    print('result_dict: ', result_dict)
    return result_dict


def compute_overlap():

    result_list  = []
    len_list = [32, 64, 128, 256, 512, 1024]

    for l in len_list:        
        result_list.append(compute_dataset(input_file = 'dataset/phrase1/training_para_1024.json', max_len = l))

    write_list_to_json_file('dataset/data_statistics.json', result_list, file_access = 'w')    
    return result_list

def instance_distribution(input_file = 'dataset/collected_data.json'):
    dataset = load_list_from_json_file(input_file, format_json = False)
    print(len(dataset))

    instance_dict =  {}
    for item in dataset:
        instances = item['instances']

        for i in instances:
            if (i[1] not in instance_dict):
                instance_dict[i[1]] = 1
            else:
                instance_dict[i[1]] += 1
 
    instance_dict = dict(sorted(instance_dict.items(), key = lambda x:x[1], reverse = True))
    write_list_to_json_file('dataset/instance_distribution.json', instance_dict, file_access = 'w')

    count_1time = 0
    for k, v in instance_dict.items():
        if (v == 1):
            count_1time += 1

    print('instances occur in 1 item: ', count_1time)
    print('instance_dict: ', len(instance_dict))

    return instance_dict, dataset


def show_instance_plot(instance_dict):

    label_list, value_list = [], []

    other_value = 0
    i = 0
    for k, v in instance_dict.items():
        if (i > 5):
            
            other_value += v
        else:
            label_list.append(k)
            value_list.append(v)
        i += 1

    label_list.append('others')
    value_list.append(other_value)

    print('label_list: ', label_list)
    print('value_list: ', value_list)

    # shown by external softwares
    

def get_training_instance(instance_dict, dataset, training_rate = 0.8):

    training_len = int(len(dataset)*training_rate)
    training_instances = []

    count = 0
    for item in instance_dict.items():
        if (count >= training_len): break
        count += item[1]
        training_instances.append(item[0])

    return training_instances


#........................................
if __name__ == "__main__":
    #compute_overlap()

    #text_by_len()
    #text_by_token()

    #instance_dict, dataset = instance_distribution()
    #show_instance_plot(instance_dict)

    #token_pos_by_para()
    #count_vocab_size()

    #check_dataset()


    


