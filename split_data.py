import random
import spacy
import pandas as pd
import os

nlp_en = spacy.load('en_core_web_md')
from read_write_file import *

def get_text_by_length(content, instance_text, max_length = 256, margin = 0):

    if (instance_text == ''):
        con_list = content.split()
        con_list = con_list[0:max_length]
        content = ' '.join(w for w in con_list)
        if (content[-1] != '.'): content += '.'
        return content
    
    con_list = content.split()
    ins_list = instance_text.split()

    con_length = max_length - margin - len(ins_list) 
    if (con_length < 0):
        con_list = content.split()
        con_list = con_list[0:max_length]
        content = ' '.join(w for w in con_list)
        if (content[-1] != '.'): content += '.'
        return content

    con_list = con_list[0:con_length]
    con_text = ' '.join(w for w in con_list)
    if (con_text[-1] != '.'): con_text += '.'
    
    ins_text = ' '.join(w for w in ins_list)

    return con_text + ' ' + ins_text
    

def split_phrase1_dataset(input_file = 'dataset/collected_data.json', format_json = False,  \
                          des_type = 'para_wd', max_length = 256, margin = 16):

    dataset = load_list_from_json_file(input_file, format_json = format_json)

    data_list = []

    for item in dataset:

       # filter bad examples
        if ('wiki' in item['label'].lower()): continue # remove Wikimedia items
        if ('wiki' in item['description'].lower()): continue # remove Wikimedia items
        if (item['first_sentence'].strip() == ''): continue # remove empty first sentences
        if (len(item['first_sentence'].split()) < 10): continue # remove short first sentences
        if (len(item['instances']) == 0): continue # remove empty instances 
        
         
        description = item['description']
        first_sentence = item['first_sentence']
        first_paragraph = item['first_paragraph']

        label = item['label']
        if (len(label) == 0): continue
        if (len(label) == 1): label = label.upper()
        else: label = label[0].upper() + label[1:]
             
        instances = item['instances']
        instances  = [i for i in instances if i[1].lower() != item['label'].lower()] # remove the label in instances

        subclasses = item['subclasses']
        subclasses  = [i for i in subclasses if i[1].lower() != item['label'].lower()] # remove the label in subclasses
        
        instance_text = '' 
        if (len(instances) == 1): instance_text = label + ' is a ' + instances[0][1] + '.'
        elif (len(instances) == 2): instance_text = label + ' is a ' + instances[0][1] +  ' and ' + instances[1][1] + '.'   
        elif (len(instances) > 1):
            instance_text = ', '.join(str(i[1]) for i in instances[0:len(instances)-1] if str(i).strip() != '')
            instance_text = label  + ' is a ' + instance_text
            instance_text += ', and ' + instances[-1][1] + '.'

        source = ''
        if (des_type == 'sen_wd'): # first sentence + wikidata instances
            #source = first_sentence
            #if (instance_text != ''):  source += ' ' + instance_text
            source = get_text_by_length(first_sentence, instance_text, max_length = max_length)
        elif(des_type == 'para_wd'): # first sentence + wikidata instances
            #source = first_paragraph
            #if (instance_text != ''):  source += ' ' + instance_text
            source = get_text_by_length(first_paragraph, instance_text, max_length = max_length)
            
        elif(des_type == 'para'): # first sentence only
            #source = first_paragraph
            con_list = first_paragraph.split()
            con_list = con_list[0:max_length - margin]
            source = ' '.join(w for w in con_list)
        
        else:
            #source = first_sentence
            con_list = first_sentence.split()
            con_list = con_list[0:max_length - margin]
            source = ' '.join(w for w in con_list)
            
        source = ' '.join(w for w in [w for w in source.split() if w.strip() != ''])
        if (source[-1] != '.'): source += '.' # more elegant

        # create candidate list

        candidate_list = []
        if (len(instances) != 0): candidate_list += [i[1] for i in instances]
        if (len(subclasses) != 0): candidate_list += [i[1] for i in subclasses]
        candidate_list = list(set(candidate_list))

        if (source != ''):
            data_list.append({'wikidata_id': item['wikidata_id'], 'label': item['label'], 'source': source, 'target': description, 'baseline_candidates':candidate_list})
    
    # split into training, validation, and test sets with ratio 8:1:1
    training_list = data_list[:(len(data_list)//10)*8]
    validation_list = data_list[(len(data_list)//10)*8:]
    
    test_list = validation_list[:(len(validation_list)//10)*5]
    validation_list = validation_list[(len(validation_list)//10)*5:]

    write_list_to_jsonl_file('dataset/phrase1/random/training_' + des_type + '_' + str(max_length) + '.json', \
                             training_list, file_access = 'w')
    write_list_to_jsonl_file('dataset/phrase1/random/validation_' + des_type + '_' + str(max_length) + '.json', \
                             validation_list, file_access = 'w')
    write_list_to_jsonl_file('dataset/phrase1/random/test_' + des_type + '_' + str(max_length) + '.json', \
                             test_list, file_access = 'w')


def split_phrase1_dataset2(training_instances, input_file = 'dataset/collected_data.json', format_json = False,  \
                          des_type = 'para_wd', max_length = 256, margin = 0):

    dataset = load_list_from_json_file(input_file, format_json = format_json)
    data_list = []

    print('training_instances: ', training_instances)

    training_list, validation_list, test_list = [], [], []
    for item in dataset:

        # filter bad examples
        if ('wiki' in item['label'].lower()): continue # remove Wikimedia items
        if ('wiki' in item['description'].lower()): continue # remove Wikimedia items
        if (item['first_sentence'].strip() == ''): continue # remove empty first sentences
        if (len(item['first_sentence'].split()) < 10): continue # remove short first sentences
        if (len(item['instances']) == 0): continue # remove empty instances 
        
         
        description = item['description']
        first_sentence = item['first_sentence']
        first_paragraph = item['first_paragraph']

        label = item['label']
        if (len(label) == 0): continue
        if (len(label) == 1): label = label.upper()
        else: label = label[0].upper() + label[1:]
             
        instances = item['instances']
        instances  = [i for i in instances if i[1].lower() != item['label'].lower()] # remove the label in instances

        #if ('Wikimedia list article' in instances): continue # pass the list articles

        subclasses = item['subclasses']
        subclasses  = [i for i in subclasses if i[1].lower() != item['label'].lower()] # remove the label in subclasses
        
        instance_text = '' 
        if (len(instances) == 1): instance_text = label + ' is a ' + instances[0][1] + '.'
        elif (len(instances) == 2): instance_text = label + ' is a ' + instances[0][1] +  ' and ' + instances[1][1] + '.'   
        elif (len(instances) > 1):
            instance_text = ', '.join(str(i[1]) for i in instances[0:len(instances)-1] if str(i).strip() != '')
            instance_text = label  + ' is a ' + instance_text
            instance_text += ', and ' + instances[-1][1] + '.'

        source = ''
        if (des_type == 'sen_wd'): # first sentence + wikidata instances
            #source = first_sentence
            #if (instance_text != ''):  source += ' ' + instance_text
            source = get_text_by_length(first_sentence, instance_text, max_length = max_length)
        elif(des_type == 'para_wd'): # first sentence + wikidata instances
            #source = first_paragraph
            #if (instance_text != ''):  source += ' ' + instance_text
            source = get_text_by_length(first_paragraph, instance_text, max_length = max_length)
            
        elif(des_type == 'para'): # first sentence only
            #source = first_paragraph
            con_list = first_paragraph.split()
            con_list = con_list[0:max_length - margin]
            source = ' '.join(w for w in con_list)
        
        else:
            #source = first_sentence
            con_list = first_sentence.split()
            con_list = con_list[0:max_length - margin]
            source = ' '.join(w for w in con_list)
            
        source = ' '.join(w for w in [w for w in source.split() if w.strip() != ''])
        if (source[-1] != '.'): source += '.' # more elegant

        # create candidate list

        candidate_list = []
        if (len(instances) != 0): candidate_list += [i[1] for i in instances]
        #if (len(subclasses) != 0): candidate_list += [i[1] for i in subclasses]
        candidate_list = list(set(candidate_list))

        
        if (source != ''):
            check = check_common_item(instances, training_instances)
            if (check == True):
                training_list.append({'wikidata_id': item['wikidata_id'], 'label': item['label'], \
                                  'source': source, 'target': description, 'baseline_candidates':candidate_list})
            else:
                validation_list.append({'wikidata_id': item['wikidata_id'], 'label': item['label'], \
                                  'source': source, 'target': description, 'baseline_candidates':candidate_list})
    
    # split 5:5
    test_list = validation_list[(len(validation_list)//10)*5:]
    validation_list = validation_list[:(len(validation_list)//10)*5]

    print('training, validation, test: ', len(training_list), len(validation_list), len(test_list))
    
    write_list_to_jsonl_file('dataset/phrase1/diff/training_' + des_type + '_' + str(max_length) + '.json', \
                             training_list, file_access = 'w')
    write_list_to_jsonl_file('dataset/phrase1/diff/validation_' + des_type + '_' + str(max_length) + '.json', \
                             validation_list, file_access = 'w')
    write_list_to_jsonl_file('dataset/phrase1/diff/test_' + des_type + '_' + str(max_length) + '.json', \
                             test_list, file_access = 'w')


def check_common_item(list1, list2):
    for i in list1:
        if (i[1] in list2):
            return True
    return False

def split_phrase1_setup(input_file = 'dataset/collected_data.json'):
    max_list = [32, 64, 128, 256, 512, 1024]
    margin_list = [0, 0, 0, 0, 0, 0]

    instance_dict, dataset = instance_distribution(input_file)
    training_instances = get_training_instance(instance_dict, dataset)

    # create folders
    if not os.path.exists('dataset/phrase1/diff'): os.makedirs('dataset/phrase1/diff')
    if not os.path.exists('dataset/phrase1/random'): os.makedirs('dataset/phrase1/random')

    for x, y in zip(max_list, margin_list):
        #split_dataset(des_type = 'para_wd', max_length = l)
        #split_dataset(des_type = 'sen_wd', max_length = l)
        #split_dataset(des_type = 'sen', max_length = l)
        split_phrase1_dataset2(training_instances, des_type = 'para', max_length = x, margin = y)
        split_phrase1_dataset(des_type = 'para', max_length = x, margin = y)


def split_phrase2_dataset(input_file = 'dataset/collected_sum.json', format_json = True, max_length = 128):
    
    dataset = load_list_from_jsonl_file(input_file)

    training_list = dataset[:(len(dataset)//10)*8]
    validation_list = dataset[(len(dataset)//10)*8:]
    
    test_list = validation_list[(len(validation_list)//10)*5:]
    validation_list = validation_list[:(len(validation_list)//10)*5]
    
    write_list_to_jsonl_file('dataset/phrase2/training_sum.json', training_list, file_access = 'w')
    write_list_to_jsonl_file('dataset/phrase2/validation_sum.json', validation_list, file_access = 'w')
    write_list_to_jsonl_file('dataset/phrase2/test_sum.json', test_list, file_access = 'w')


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

    return instance_dict, dataset


def get_training_instance(instance_dict, dataset, training_rate = 0.8):

    training_len = int(len(dataset)*training_rate)
    training_instances = []

    count = 0
    for item in instance_dict.items():
        if (count >= training_len): break
        count += item[1]
        training_instances.append(item[0])

    return training_instances

#.........................................
if __name__ == "__main__":

    split_phrase1_setup()

    
