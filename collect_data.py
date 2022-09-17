from wiki_core import *
from read_write_file import *

import gc
import re
from random import randrange
import sys
sys.setrecursionlimit(10**6)

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

def write_index(index, file = 'dataset/index.txt'):
    write_to_text_file(file, index)

def check_index(index, file = 'dataset/index.txt'):

    index_list = []

    try:
        index_list = read_list_from_text_file(file)
    except:
        pass
    
    if (index in index_list): return True
    return False
    
def collect_single(output_file = 'dataset/collect_data.json'):

    result_dict = {}
    wikidata_id = ''
    
    while(True):

        number_id = randrange(99000001) # there are over 99 millions items
        wikidata_id = 'Q' + str(number_id)
        
        try: result_dict = get_wikidata_by_wikidata_id(wikidata_id)
        except: pass

        print('wikidata_id: ', result_dict['wikidata_id'])
        
        if not result_dict:
            write_index(wikidata_id)
            continue

        
        des = result_dict['description'].lower().strip()

        # remove empty spaces
        result_dict['first_sentence'] = ' '.join(w for w in [w for w in result_dict['first_sentence'].split() if w.strip() != ''])
        result_dict['first_paragraph'] = ' '.join(w for w in [w for w in result_dict['first_paragraph'].split() if w.strip() != ''])
        
        sen = result_dict['first_sentence'].strip()
        
        if (des != '' and 'wiki' not in des and sen != '.' and sen != ''):
            if (check_index(wikidata_id) == False):
                break
            
        write_index(wikidata_id)
        

    print('description: ', result_dict['description'])
    print('first_sentence: ', result_dict['first_sentence'])
    print('first_paragraph: ', result_dict['first_paragraph'])
    print('---------------------------------------')
    print('---------------------------------------')

    write_single_dict_to_json_file(output_file, result_dict)
    write_index(wikidata_id)
    gc.collect()

def collect_multi(max_workers = 8, limit = 1000000, output_file = 'dataset/collected_data.json'):

    number_list = [i for i in range(1, limit + 1)]

    with ThreadPoolExecutor(max_workers = max_workers) as executor:
        results = executor.map(collect_single, [output_file]*len(number_list), timeout = 600)

#........................................
if __name__ == "__main__":
    collect_multi()
    

    #print(get_wikidata_by_wikidata_id('Q5109413'))
    '''content  = 'John F. Kennedy (May 29, 1917 .... – November 22, 1963), often referred to by his initials as JFK or by the nickname Jack, was an American politician who served as the 35th president of the United States from 1961 until his assassination near the end of his third year in office.'
    first_sen = extract_first_sentence(content)
    print(first_sen)'''
