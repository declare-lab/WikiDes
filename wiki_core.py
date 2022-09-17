import csv
import gc
import os
import subprocess
import re
import requests
import xml.etree.ElementTree as ET
import json
import pandas as pd

import nltk
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
import urllib

import sys
sys.setrecursionlimit(3000)

from datetime import *
from dateutil.easter import *
from dateutil.parser import *
from dateutil.relativedelta import *
from dateutil.rrule import *

from read_write_file import *

boundary = re.compile('^[0-9]$')
tag_re = re.compile(r'<[^>]+>')

import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_md')
nlp.add_pipe(nlp.create_pipe('sentencizer'), before='parser')

import sys
non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), '')

# remove emojis, function is from Karim Omaya (stackoverflow.com)
def remove_emojis(data):
    emoj = re.compile('['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002500-\U00002BEF'  # chinese char
        u'\U00002702-\U000027B0'
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        u'\U0001f926-\U0001f937'
        u'\U00010000-\U0010ffff'
        u'\u2640-\u2642' 
        u'\u2600-\u2B55'
        u'\u200d'
        u'\u23cf'
        u'\u23e9'
        u'\u231a'
        u'\ufe0f'  # dingbats
        u'\u3030'
                      ']+', re.UNICODE)
    return re.sub(emoj, '', data)

# clear screen
def cls():
    os.system('cls' if os.name == 'nt' else 'clear')

# get xml data by page title (English Wikipedia)
def get_xml_data_by_title(title):
    link = 'https://en.wikipedia.org/w/api.php?action=query&redirects&format=xml&rvprop=content&prop=extracts|revisions|pageprops|templates|categories&rvslots=main&titles=' + urllib.parse.quote(title)
    response = requests.get(link, timeout = 30) # 30s
    root = ET.fromstring(response.text)
    return root

def get_first_paragraph(title):
    link = 'https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&titles=' + urllib.parse.quote(title)
    response = requests.get(link, timeout = 30) # 30s

    first_paragraph = ''
    #print('link: ', link)
    #print('response.text: ', response.text)
    try:
        data_dict = json.loads(response.text)
        data_dict = data_dict['query']['pages']
        key = next(iter(data_dict)) 
        first_paragraph = data_dict[key]['extract']
        
    except Exception as e:
        #print('Error --- get_first_paragraph: ', e)
        pass

    return first_paragraph


def get_triple(word, level = 2):

    #print('word: ', word)

    triple_list = []
    term_dict = get_wikidata_by_text(word)
    if not term_dict: return []
    
    instances = term_dict['instances']
    subclasses = term_dict['subclasses']
    parts = term_dict['parts']
            
    for i in instances: triple_list.append([term_dict['label'], 'instance_of', i[1]])
    for s in subclasses: triple_list.append([term_dict['label'], 'subclass_of', s[1]])
    for p in parts: triple_list.append([term_dict['label'], 'part_of', p[1]])

    # get property data
    triple_list += get_extend_triple(triple_list, instances, subclasses, parts, level)

    return triple_list

def get_extend_triple(triple_list, instances, subclasses, parts, level):

    if (level == 0):
        return triple_list
    
    # get property data
    id_list = [i[0] for i in instances]
    id_list += [s[0] for s in subclasses]
    id_list += [p[0] for p in parts]

    for item in id_list:
        term_dict = get_wikidata_by_wikidata_id(item)
        try: 
            instances = term_dict['instances']
            for i in instances: triple_list.append([term_dict['label'], 'instance_of', i[1]])
        except Exception as e:
            print('Error1 -- get_extend_triple: ', e)
            instances = {}
            pass

        try:
            subclasses = term_dict['subclasses']
            for s in subclasses: triple_list.append([term_dict['label'], 'subclass_of', s[1]])
        except Exception as e:
            print('Error2 -- get_extend_triple: ', e)
            subclasses = {}
            pass

        try:
            parts = term_dict['parts']
            for p in parts: triple_list.append([term_dict['label'], 'part_of', p[1]])
        except Exception as e:
            print('Error3 -- get_extend_triple: ', e)
            parts = {}
            pass

    level = level - 1
    #print(triple_list, instances, subclasses, parts, level)
    
    return get_extend_triple(triple_list, instances, subclasses, parts, level)
    

def get_wikidata_by_wikidata_id(wikidata_id):

    result_dict = {}

    try:
        wikidata_root = get_wikidata_root(wikidata_id)
        description = get_description(wikidata_root)
        #print('description: ', description)
        #print('wikidata_id: ', wikidata_id)
    
        if ('disambiguation page' in description.lower() or 'wiki' in description.lower()): return {}

        label = get_label(wikidata_root)
        claims = get_claims(wikidata_root, wikidata_id)
        instances = get_instance_of(claims)
        subclasses = get_subclass_of(claims)
        #parts = get_part_of(claims)
        aliases = get_alias(wikidata_root)

        result_dict['wikidata_id'] = wikidata_id
        result_dict['label'] = label
        result_dict['description'] = description
        result_dict['instances'] = instances
        result_dict['subclasses'] = subclasses
        result_dict['aliases'] = aliases

        # get the first paragraph & first sentence of Wikipedia
        sitelink = get_sitelink(wikidata_root)
        #wiki_root = get_xml_data_by_title(sitelink)
        first_paragraph = get_first_paragraph(sitelink)
        sents = sent_detector.tokenize(first_paragraph)

        #print('sents: ', sents)

        result_dict['first_paragraph'] = ' '.join(s for s in sents)
        result_dict['first_sentence'] = sents[0]
        
        
    except Exception as e:
        #print('Error --- get_wikidata_by_wikidata_id: ', e)
        pass
    
    return result_dict
    

def get_wikidata_by_text(title):

    result_dict = {}
    root = get_xml_data_by_title(title)
    
    wikidata_id = get_wikidata_id(root)
    if (wikidata_id == '' or wikidata_id == None): return {}

    wikidata_root = get_wikidata_root(wikidata_id)
    description = get_description(wikidata_root)
    if ('disambiguation page' in description): return {}

    label = get_label(wikidata_root)

    claims = get_claims(wikidata_root, wikidata_id)
    instances = get_instance_of(claims)
    subclasses = get_subclass_of(claims)
    parts = get_part_of(claims)

    result_dict['wikidata_id'] = wikidata_id
    result_dict['label'] = label
    result_dict['description'] = description
    result_dict['instances'] = instances
    result_dict['subclasses'] = subclasses
    result_dict['parts'] = parts
    
    return result_dict

# get XML's root of a wiki page by its wikidataID
def get_wikidata_root(wikidata_id):
    if (wikidata_id == None):
        wikidata_id = ''
    link = 'https://www.wikidata.org/w/api.php?action=wbgetentities&format=xml&ids=' + wikidata_id
    root = get_xml_data_by_url(link)
    if (root is None):
        return ''
    return root
    
# get property's datatype
def get_property_datatype(root):
    try:
        for x in root.find('./entities'):
            #print(x)
            value = remove_emojis(x.attrib['datatype'])
            if (value != ''):  
                return value
    except:
        pass
    return ''                

# get xml data by url
def get_xml_data_by_url(link):
    response = requests.get(link, timeout = 30)
    root = ET.fromstring(response.text)
    return root

# get English Wikipedia (enwiki) sitelink
def get_sitelink(root):
    try:
        for x in root.find('./entities/entity/sitelinks'):
            if (x.attrib['site'] == 'enwiki'):
                value = remove_emojis(x.attrib['title'])
                if (value != ''):  
                    return value
    except:
        pass
    return ''

# get wikidataID
def get_wikidata_id(root):
    for node in root.find('./query/pages/page'):
        if (node.tag == 'pageprops'):
            return node.attrib['wikibase_item']
    return ''

# get article content in HTML format
def html_content(root):
    text = ''
    for x in root.iter('extract'):
        text += x.text
    text = text.replace('\n',' ')    
    return text

# split content into sections, not use for h3, h4, h5
def get_content_by_section(text):
    secs = re.split('<h2>(.*?)</h2>', text)
    secs = [tag_re.sub(r' ', x) for x in secs] 
    secDict = dict()
    key = 'Beginning' # first part
    i = 0
    for x in secs:
        if (i%2 == 0):
            secDict[key] = x.strip()
        else:
            key = x
        i = i + 1
    return dict1

# get text without sections
def get_text_not_by_section(text):
    #print('raw text: ', text)
    list_headers = re.findall('<(h1|h2|h3|h4|h5|h6)>(.*?)</(h1|h2|h3|h4|h5|h6)>', text)
    
    for x in list_headers:
        sub = ''
        try:
            sub += '<' + x[0] + '>' + x[1] + '</' + x[2] + '>'
            #print(sub)
            text = text.replace(sub, '')
        except:
            continue

    text = re.sub(tag_re, '', text)
    #print('text:', text)
    return text


def extract_first_sentence(text):

    count = 0
    first_sentence = ''
    for w in text:
        first_sentence += w

        if (w == '('): count += 1
        if (w == ')'): count -= 1

        if (w == '.'):        
            if (len(first_sentence.split()) > 10 and count == 0): break

        #print('----', first_sentence)

    return  first_sentence


# count sentences
def sentence_list(text):
    text = text.replace(u'\xa0', u' ') # remove non-breaking space \xa0
    text = text.replace(u'"', u'') # remove "

    '''text = re.sub('\(.*?\)','', text) # remove all content in rounded brackets
    text = text.replace('(', '')
    text = text.replace(')', '')'''
    
    sentences = text.split('.')    # really wrong errors
    sentences = [x.strip() for x in sentences]
    sentences = [x + '.' for x in sentences]

    
    return sentences

# split sentences by spaCy sentenizer
def sentence_list_sentencizer(text):
    text = text.replace(u'\xa0', u' ') # remove non-breaking space \xa0
    text = text.replace(u'"', u'') # remove "
 
    doc = nlp(text)

    sents_list = []
    for sent in doc.sents:
        sents_list.append(sent.text.strip())
           
    return sents_list

# get label (page name) of a Wiki page by its wikidataID
def get_label_by_wikidataID(wikidataID):
    link = 'https://www.wikidata.org/w/api.php?action=wbgetentities&format=xml&ids=' + wikidataID
    root = get_xml_data_by_url(link)           
    return get_label(root)

# get label (page name) by XML's root
def get_label(root):
    try:
        for x in root.find('./entities/entity/labels'):
            if (x.attrib['language'] == 'en-gb' or x.attrib['language'] == 'en'):
                value = remove_emojis(x.attrib['value'])
                if (value != ''):
                    return value
    except:
        pass
    return ''

# get a short description of a Wiki page by XML's root
def get_description(root):

    try:
        for x in root.find('./entities/entity/descriptions'):
            if (x.attrib['language'] == 'en-gb' or x.attrib['language'] == 'en'):
                value = remove_emojis(x.attrib['value'])
                if (value != ''):  
                    return value
    except:
        pass
    return ''      

# get values by property
def get_values_by_property(claims, property_name):
    result_list = []
    try:
        for c in claims:
            if (c[1][1] == property_name):
                k = c[1][3]
                root = get_wikidata_root(k)
                label = get_label(root)
                #wikidata_id = get_wikidata_id(root)
                result_list.append([k, label])
    except:
        pass

    return result_list

def get_instance_of(claims):
    result_list = []
    try:
        for c in claims:
            if (c[1][1] == 'P31'):
                k = c[1][3]
                root = get_wikidata_root(k)
                instance_name = get_label(root)
                #wikidata_id = get_wikidata_id(root)
                result_list.append([k, instance_name])
    except:
        pass

    return result_list
    
# get subclass of (P279)
def get_subclass_of(claims):
    result_list = []
    try:
        for c in claims:
            if (c[1][1] == 'P279'):
                k = c[1][3]
                root = get_wikidata_root(k)
                instance_name = get_label(root)
                #wikidata_id = get_wikidata_id(root)
                result_list.append([k, instance_name])
    except:
        pass

    return result_list

# get part of (P361)
def get_part_of(claims):
    result_list = []
    try:
        for c in claims:
            if (c[1][1] == 'P361'):
                k = c[1][3]
                root = get_wikidata_root(k)
                instance_name = get_label(root)
                #wikidata_id = get_wikidata_id(root)
                result_list.append([k, instance_name])
    except:
        pass

    return result_list
    

# get nationality (P27 - country of citizenship)
def get_nationality(claims):

    result_list = []
    try:
        for c in claims:
            if (c[1][1] == 'P27'):
                k = c[1][3]
                root = get_wikidata_root(k)
                country_name = get_label(root)
                result_list.append([k, country_name])
    except:
        pass

    return result_list
        
# get alias (other label names) of a Wiki page by XML's root
def get_alias(root):
    aliases = []
    try:
        for z in root.find('./entities/entity/aliases'):
            if (z.attrib['id'] == 'en-gb' or z.attrib['id'] == 'en'):
                for t in z:
                    value = remove_emojis(t.attrib['value'])
                    if (value != ''): 
                        aliases.append(value)
    except:
        pass
    return aliases
        
# get claims (Wikidata's statements) of a Wiki page   
def get_claims(root, wikidataID):

    #print('root, wikidataID:', root, wikidataID)

    claim_list = [] # statement list

    if (root == '' or root is None):
        return claim_list

    if (wikidataID == '' or wikidataID is None):
        return claim_list

    s = wikidataID # s: subject (item identifier, wikidataID)
    p = ob = pt = pv = q = qt = qv = ''
    # p: predicate (property), ob: object (property value identifier)
    # pt: object type (property value type), pv: object value (property value)
    # q: qualifier, qt: qualifier type, qv: qualifier value

    # loop each predicate (property)
    for predicate in root.find('./entities/entity/claims'):        
        #print('************************')
        #print('Property: ', predicate.attrib['id'])
        p = remove_emojis(predicate.attrib['id']) # predicate (property)
        for claim in predicate.iter('claim'):
            pt = remove_emojis(claim[0].attrib['datatype']) # property type
            #print('+', pt)
            for obj in claim.find('mainsnak'):
                try:
                    try:
                        # obj.attrib['value'].encode('unicode-escape').decode('utf-8')
                        pv = remove_emojis(obj.attrib['value'])
                    except Exception as e:
                        #print('Error:', e)
                        pass
                    if (pv != ''):
                        continue
                    objdict = obj[0].attrib

                    if ('id' in objdict):
                        #print('--', objdict['id'])
                        ob = remove_emojis(objdict['id']) # qualifier
                    elif ('time' in objdict):
                        #print('--', objdict['time'])
                        pv = remove_emojis(objdict['time']) # time
                    elif ('amount' in objdict):
                        #print('--', objdict['amount'])
                        pv = remove_emojis(objdict['amount']) # amount
                    # capture other data types (globle coordinate, etc)
                    # ...
                    else:
                        pass
                        #print('--', 'empty')
                except Exception as e:
                    pass
                    #print('Error:', e) 
			
            # check the number of qualifiers
            qual_properties = [t for t in claim.findall('qualifiers/property')]
            if (len(qual_properties) == 0):
                if (pt != 'wikibase-item'):
                    r1 = [s, p, pt, pv]
                    claim_list.append(['r1', r1]) # WST-1 statement
                else:
                    r2 = [s, p, pt, ob]
                    claim_list.append(['r2', r2]) # WST-2 statement
            else:
                if (pv != ''):
                    r3 = [s, p, pt, pv] # WST3-a
                else:
                    r3 = [s, p, pt, ob] # WST3-b 
                try:
                    for x in claim.find('qualifiers'):
                        #print('----', x.attrib['id'], x.tag)
                        q = remove_emojis(x.attrib['id']) # qualifier identifier
                        qt = remove_emojis(x[0].attrib['datatype']) # qualifier data type
                        subr = [q, qt]
                        children = x.getchildren()
                        for y in children:
                            for z in y.find('datavalue'):
                                qv = '' # qualifier value
                                if ('id' in z.attrib):
                                    #print('--------', z.attrib['id'])
                                    qv = remove_emojis(z.attrib['id']) # qualifier value
                                elif ('time' in z.attrib):
                                    #print('--------', z.attrib['time'])
                                    qv = remove_emojis(z.attrib['time']) # value
                                elif ('amount' in z.attrib):
                                    #print('--------', z.attrib['amount'])
                                    qv = remove_emojis(z.attrib['amount']) # value
                                # capture other data types (globle coordinate, etc)
                                # ...   
                                else:
                                    #print('--------', 'empty')
                                    qv = '' # set to empty
                                if (qv != ''):
                                    subr.append(qv)
                                    r3.append(subr) # add a qualifier value
                                    qv = '' # set to empty for new iterator
                except Exception as e:
                    pass
                    #print('Error: ', e)
                
                if (len(r3) > 4):
                    claim_list.append(['r3', r3]) # WST-3 statement
                else:
                    if (pt != 'wikibase-item'):
                        claim_list.append(['r1', r3]) # WST-1 statement
                    else:
                        claim_list.append(['r2', r3]) # WST-2 statement
            ob = pv = '' # reset values (important)
        #print('************************')    

    '''for c in claim_list:
        print('-----------------')
        print(c)'''

    return claim_list     

# get wikidata item
def get_wikidata_item_by_name(item_name, file_name, depth):

    item_name = item_name.replace(' ','_') # format page name
    root = ''
    wikidata_id = ''
    
    try:
        root = get_xml_data_by_title(item_name)
        wikidata_id = get_wikidata_id(root)
    except Exception as e:
        #print('Error:', e)
        return []

    if (wikidata_id == ''):
        return []

    if (check_exist_in_item_list(wikidata_id) == True):
        return

    if (depth == 0):
        return []

    #print(wikidata_id)
    wikidata_root = get_wikidata_root(wikidata_id)
    xmlstr = ET.tostring(wikidata_root, encoding='unicode', method='xml')

    items = []
    items = match_wikidata_item(xmlstr)
    
    if ('Could not find an entity' in xmlstr or 'missing=""' in xmlstr):
        return

    label = description = ''
    alias = claims = []

    try:
        label = get_label(wikidata_root)
        description = get_description(wikidata_root)
        alias = get_alias(wikidata_root)
        claims = get_claims(wikidata_root, wikidata_id)
    except Exception as e:
        #print('Error:', e)
        return []
        
    write_to_text_file('item_id_list.txt', wikidata_id)
    write_wikidata_to_csv_file(file_name, wikidata_id, label, description, alias, claims)
	
    for i in items:
        get_wikidata_item_by_id(i, file_name, depth-1)

# get wikidata item
def get_wikidata_item_by_id(wikidata_id, file_name, depth):

    if (check_exist_in_item_list(wikidata_id) == True):
        return []

    if (depth == 0):
        return []

    wikidata_root = get_wikidata_root(wikidata_id)
    xmlstr = ET.tostring(wikidata_root, encoding='unicode', method='xml')

    items = []
    items = match_wikidata_item(xmlstr)
    
    if ('Could not find an entity' in xmlstr or 'missing=""' in xmlstr):
        return []

    label = description = ''
    alias = claims = []

    try:
        label = get_label(wikidata_root)
        description = get_description(wikidata_root)
        alias = get_alias(wikidata_root)
        claims = get_claims(wikidata_root, wikidata_id)
        
    except Exception as e:
        #print('Error:', e)
        return []

    write_to_text_file('item_id_list.txt', wikidata_id)
    write_wikidata_to_csv_file(file_name, wikidata_id, label, description, alias, claims)
    for i in items:
        get_wikidata_item_by_id(i, file_name, depth-1)
    
# get other items appearing in an item structure
def match_wikidata_item(text):
    items = []
    try:
        items = re.findall(r'"Q\d+"', text)
        items = list(set([i[1:-1] for i in items]))
    except Exception as e:
        pass
        #print('Error:', e)
    return items

# check exist in item list
def check_exist_in_item_list(item_id):
    item_list = read_from_text_file('item_id_list.txt')
    if (item_id in item_list):
        return True
    return False

# get Wikidata hypernyms by level
def get_hypernyms(values, results, level):

    #print('values: ', values)
    #print('results: ', results)
    #print('~~~~~~~~~~~~~~~~~~~~~~')
    
    if (level == 0 or len(values) == 0):
        results += values
        results = [list(x) for x in set(tuple(x) for x in results)] # keep unique values
        return results

    terms = []
    for v in values:
        try:
            root = get_wikidata_root(v[0])
            claims = get_claims2(root, v[0])

            terms += get_values_by_property(claims, 'P31')
            terms += get_values_by_property(claims, 'P279')
        except:
            pass
        
    results += values
    results += terms
    results = [list(x) for x in set(tuple(x) for x in results)] # keep unique values

    terms = set(tuple(x) for x in terms)
    values = set(tuple(x) for x in values)
    terms = terms - values
    terms  = [list(x) for x in terms]

    #print('terms: ', terms)
    return get_hypernyms(terms, results, level - 1)

# check item in a list, return an entity
def get_item_entities(item, entities):
    for e in entities:
        try:
            if (item[0] == e[0] and item[1] == e[1] and item[2] == e[2]):
                return e
        except:
            pass
    return []

# get list by category name (future development)
def get_page_list_by_category(category_name):
    return

# map page name list to wikidata and write results to output file
def map_page_list_to_wikidata(input_file, output_file):

    #input_file = 'list.txt'
    page_list = read_from_text_file(input_file)
    
    for p in page_list:
        try:

            wikidata_id = ''
            try:
                root = get_xml_data_by_title(p)
                wikidata_id = get_wikidata_id(root)
            except:
                pass

            wikidata_root = get_wikidata_root(wikidata_id)
            claims = get_claims(wikidata_root, wikidata_id)
            #print(p)
        
            try:
                countries = get_nationality(claims)
                country_string = ';'.join(e[1].strip() for e in countries) 
            except:
                pass
                
            try:
                aliases = get_alias(wikidata_root)
                alias_string = ';'.join(e.strip() for e in aliases)
            except:
                pass

            try:
                description = get_description(wikidata_root)
            except:
                pass

            write_wikidata_to_csv_file2(output_file, wikidata_id, p, description, alias_string, country_string)
        except Exception as e:
            #print(e)
            pass

# scan items by page list
def scan_items_from_page_list(file_name, depth):
    page_list = read_from_text_file(file_name)
    for p in page_list:
        #print(p)
        get_wikidata_item_by_name(p, file_name, depth)

# scan items by id range (default id_type is wikidata item)
def scan_items_from_id_range(id_type, start, stop, depth):

    prefix = 'Q'
    if (id_type == 'property'):
        prefix = 'P'
        
    for item_id in range(start, stop):
        item_id = prefix + str(item_id)
        #print(item_id)
        get_wikidata_item_by_id(item_id, file_name, depth)

def scan_people_from_list(file_name, output_file):
    page_list = read_from_text_file(file_name)

    for p in page_list:

        root = wikidata_id = wikidata_root = ''

        try:
            root = get_xml_data_by_title(p)
            wikidata_id = get_wikidata_id(root)
            wikidata_root = get_wikidata_root(wikidata_id)
        except:
            pass

        if (wikidata_id == '' or wikidata_id == None): continue

        label = get_label(wikidata_root)
        
        claims = get_claims(wikidata_root, wikidata_id)
        instances = get_instance_of(claims)
        instances = ';'.join(e[1].strip() for e in instances)
        
        '''aliases = get_alias(wikidata_root)
        alias_string = ';'.join(e.strip() for e in aliases)
        description = get_description(wikidata_root)'''

        write_wikidata_to_csv_file3(output_file, wikidata_id, label, instances)


# update to blazegraph from item list
def update_to_blaze_graph(file_name, IP):

    page_list = read_from_csv_file(file_name, '#', 'all')

    
    for p in page_list:
        print(p[0])
        command = "curl -X POST http://" + IP + "/blazegraph/sparql --data-urlencode 'update=DROP ALL; LOAD <https://www.wikidata.org/entity/" + p[0] + ".ttl>;'"
        print(command)
        os.system(command)
        #os.system('cmd \c "color a & date"')
        #subprocess.call(command, creationflags = 0x08000000)

def quote_authors_by_country(file_name, output_file):

    page_list = read_from_csv_file(file_name, '#', 'all')

    nationalities = ['Indonesia','Vietnam', 'Thailand', 'Brunei', 'Laos',
                     'Cambodia', 'Myanmar', 'Malaysia', 'Philippines', 'East Timor']

    '''for p in page_list:

        if (p'''
        

    dtypes = {
    "wikidata_id": "category",
    "label": "category",
    "nationalities": "category",
    "description": "category",
    "aliases": "category",
    "quote_author": "category",
    "verb": "category",
    "object": "category",
    "quote": "category",
    "quote_sentence": "category",
    "url": "category",
    "news_title": "category",
    "date": "category",
    "update_date": "category",
    "publisher": "category",
    "news_author": "category"
    }
 
    #df = pd.read_csv(file_name, dtype=dtypes,usecols=list(dtypes))
    #quotes = df.loc[df['quote'].str.contains('Singapore')]
    #print(quotes.to_string())


    # by nationalities
    '''group_by_nationalities = df.groupby("nationalities")["label"]
    data_list = []
    for x, y in group_by_nationalities:
        #print(x, '---', y.count())
        data_list.append([x, y.count()])

    data_list = sorted(data_list, key = lambda x :x[1])
    for d in data_list:
        data = str(d[0]) + ',' + str(d[1])
        write_to_text_file(output_file, data)'''

    # by unique quote authors
    '''by_wikidata_id = df.groupby("wikidata_id")["label"]
    #by_wikidata_id.describe()
    data_list = []
    for x, y in by_wikidata_id:
        #print(x, '---', y.iloc[0])
        data_list.append([x, y.iloc[0]])

    data_list = sorted(data_list, key = lambda x: str(x[1]))
    for d in data_list:
        data = str(d[0]) + ',' + str(d[1])
        write_to_text_file(output_file, data)'''
  
#.............................................................................
    
#output_file_name = 'list_page.csv'
#Q148,China
#Q252,Indonesia
#Q928,Philippines
#get_wikidata_item_by_id('Q928', output_file_name, 3)
#get_wikidata_item_by_name('1971 Indonesian legislative election', output_file_name, 3)
#get_wikidata_item_by_name('Simone Loria', output_file_name, 1)
#scan_items_from_page_list(output_file_name, 3)
#scan_people_from_list('data\Indonesian.txt', 'data\Indonesian_people.txt')

#curl -X POST http://172.29.0.1:9999/blazegraph/sparql --data-urlencode 'update=DROP ALL; LOAD <https://www.wikidata.org/entity/Q222.ttl>;'

#update_to_blaze_graph('data\wikidata_items.csv', '192.168.0.13:9999')

#quote_authors_by_country('data/quotes.csv','by_wikidata_id.csv')


