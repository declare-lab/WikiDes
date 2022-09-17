import pandas as pd
import krippendorff
import torch

# https://www.nltk.org/_modules/nltk/metrics/agreement.html
from nltk import agreement
from nltk.metrics.distance import masi_distance
from nltk.metrics.distance import jaccard_distance

from metrics import *
from read_write_file import *

def create_annot_data(n_items = 50, input_file = 'dataset\phrase2\generated_test_para_256_diff.json', \
                      output_file = 'dataset\human_eval.json'):

    dataset = load_list_from_jsonl_file(input_file)
    ouput_list = []

    for item in dataset:

        if (len(ouput_list) > n_items): break

        source = item['source']
        target = item['target']
        candidates = item['candidate']

        try: candidates.remove(target)
        except: pass
        if (len(candidates) == 0): continue

        best_can = ''
        best_score = 0

        print('source: ', source)

        for can in candidates:
            output = compute_rouge_single(can, source)
            print('-- can: ', can)
            score = output['rouge1_fmeasure']

            if (score > best_score):
                best_score = score
                best_can = can

        print('-- best can: ', best_can)
        print('--------------------------')
        ouput_list.append({'source':source, 'sum1':best_can, 'sum2':target})

    write_list_to_jsonl_file(output_file, ouput_list, file_access = 'a')



def load_file(file_list, field = 'best_summary'):

    data_dict = {}
    
    for index, f in enumerate(file_list):

        temp_list = load_list_from_jsonl_file(f)
        temp_list = [item['coder'] for item in temp_list]
        
        data_dict['coder' + str(index + 1)] = temp_list

    #print('data_dict: ', data_dict, len(data_dict))

    # convert to csv

    coder1 = data_dict['coder1']
    coder2 = data_dict['coder2']
    coder3 = data_dict['coder3']

    data_list = []
    for i, c1, c2, c3 in zip(range(1, len(coder1) + 1), coder1, coder2, coder3):

        item_dict = {'index':i, 'coder1':c1[field], 'coder2':c2[field], 'coder3':c3[field]}
        data_list.append(item_dict)

    data_list.insert(0, {'index':'index', 'coder1':'coder1', 'coder2':'coder2', 'coder3':'coder3'})

    output_file = 'dataset/human_eval_' + field + '.csv'
    write_list_to_tsv_file(output_file, data_list, delimiter = ',', file_access = 'w')
        
    return data_dict
        

def calculate_agreement(input_file = 'dataset/human_eval.csv'):

    result_dict = {}
    
    # read file
    df = pd.read_csv(input_file)

    # krippendorff ---------------
    matrix = []
    for index, row in df.iterrows():
        matrix.append(list(row)[1:]) # remove first column

    matrix = torch.tensor(matrix).T
    matrix_flatten = torch.flatten(matrix)
    matrix_flatten = matrix_flatten.tolist()
    result_dict['average'] = sum(matrix_flatten)/len(matrix_flatten)

    dis_dict = {}
    for i in matrix_flatten:
        if (i not in dis_dict): dis_dict[i] = 1
        else: dis_dict[i] += 1
            
    result_dict['distribution'] = dis_dict


    # "nominal", "ordinal" (memory error), "interval"
    alpha_nominal = krippendorff.alpha(matrix, level_of_measurement="nominal")
    alpha_interval = krippendorff.alpha(matrix, level_of_measurement="interval")
    result_dict['alpha_nominal'] = alpha_nominal
    result_dict['alpha_interval'] = alpha_interval
    # -----------------------------

    # NLTK ------------------------
    annots = []
    for idx, row in df.iterrows():
        annot_id = idx
        annot_coder1 = ['coder1', annot_id, frozenset(str(row.coder1))]
        annot_coder2 = ['coder2', annot_id, frozenset(str(row.coder2))]
        annot_coder3 = ['coder3', annot_id, frozenset(str(row.coder3))]
        annots.append(annot_coder1)
        annots.append(annot_coder2)
        annots.append(annot_coder3)

    jaccard_task = agreement.AnnotationTask(distance=jaccard_distance)
    masi_task = agreement.AnnotationTask(distance=masi_distance)
    tasks = [jaccard_task, masi_task]
    for task in tasks:
        task.load_array(annots)
        '''print("Statistics for dataset using {}".format(task.distance))
        print("C: {}\nI: {}\nK: {}".format(task.C, task.I, task.K))
        print("Pi: {}".format(task.pi()))
        print("Kappa: {}".format(task.kappa()))
        print("Multi-Kappa: {}".format(task.multi_kappa()))
        print("Alpha: {}".format(task.alpha()))'''

        if (task is jaccard_task):
            result_dict['pi_jaccard'] = task.pi()
            result_dict['s_jaccard'] = task.S()
            result_dict['kappa_jaccard'] = task.kappa()
            result_dict['multi-kappa_jaccard'] = task.multi_kappa()
        else:
            result_dict['pi_masi'] = task.pi()
            result_dict['s_masi'] = task.S()
            result_dict['kappa_masi'] = task.kappa()
            result_dict['multi-kappa_masi'] = task.multi_kappa()
    # -----------------------------

    print('result_dict: ', result_dict)
    print('---------------------------------')
    return result_dict

#...........................................
'''
create_annot_data(n_items = 50, input_file = 'dataset\phrase2\generated_test_para_256_diff.json', \
                      output_file = 'dataset\human_eval.json')
create_annot_data(n_items = 50, input_file = 'dataset\phrase2\generated_test_para_256_random.json', \
                      output_file = 'dataset\human_eval.json')
'''


file_list = ['dataset/human_eval_output_1.json', 'dataset/human_eval_output_2.json', 'dataset/human_eval_output_3.json']
load_file(file_list, field = 'best_summary')
load_file(file_list, field = 'adequacy')
load_file(file_list, field = 'relevance')
load_file(file_list, field = 'correctness')
load_file(file_list, field = 'concise')


calculate_agreement('dataset/human_eval_best_summary.csv')
calculate_agreement('dataset/human_eval_adequacy.csv')
calculate_agreement('dataset/human_eval_relevance.csv')
calculate_agreement('dataset/human_eval_correctness.csv')
calculate_agreement('dataset/human_eval_concise.csv')

