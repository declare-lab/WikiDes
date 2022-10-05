# rouge
#from rouge_score import rouge_scorer # native rouge package

# moverscore
#from moverscore_v2 import get_idf_dict, word_mover_score 
#from collections import defaultdict

# other metrics (BLEU, BertScore, etc)
import datasets

meteor = datasets.load_metric('meteor')
bleu = datasets.load_metric('bleu')
bertscore = datasets.load_metric('bertscore')

def compute_rouge_batch(predictions, references):

    """
        predictions: list
        references: list
    """

    result_dict = {}

    predictions = list(predictions)
    references = list(references)
    
    if (type(predictions) != list or type(references) != list):
        print('"predictions" or "references" is not a list!')
        return result_dict

    scorer = datasets.load_metric('rouge')
        
    r1_pre, r2_pre, rL_pre = [], [], []
    r1_re, r2_re, rL_re = [], [], []
    r1_fm, r2_fm, rL_fm = [], [], []
        
    for pre, ref in zip(pre, ref):
        output = compute_rouge_single(pre, ref)
        r1_pre.append(output['rouge1_precision'])
        r2_pre.append(output['rouge2_precision'])
        rL_pre.append(output['rougeL_precision'])

        r1_re.append(output['rouge1_recall'])
        r2_re.append(output['rouge2_recall'])
        rL_re.append(output['rougeL_recall'])

        r1_fm.append(output['rouge1_fmeasure'])
        r2_fm.append(output['rouge2_fmeasure'])
        rL_fm.append(output['rougeL_fmeasure'])
                
    result_dict['rouge1_precision'] = r1_pre
    result_dict['rouge2_precision'] = r2_pre
    result_dict['rougeL_precision'] = rL_pre
        
    result_dict['rouge1_recall'] = r1_re
    result_dict['rouge2_recall'] = r2_re
    result_dict['rougeL_recall'] = rL_re
        
    result_dict['rouge1_fmeasure'] = r1_fm
    result_dict['rouge2_fmeasure'] = r2_fm
    result_dict['rougeL_fmeasure'] = rL_fm  

    #print('result_dict: ', result_dict)
    return result_dict
    

def compute_rouge_single(prediction, reference):
    """
        predictions: single string
        references: single string
    """

    result_dict = {}
    
    if (type(prediction) != str or type(reference) != str):
        print('"prediction" or "reference" is not a string!')
        return result_dict
    
    scorer = datasets.load_metric('rouge')
    output = scorer.compute(predictions=[prediction], references=[reference])
        
    result_dict['rouge1_precision'] = output['rouge1'].mid.precision
    result_dict['rouge2_precision'] = output['rouge2'].mid.precision
    result_dict['rougeL_precision'] = output['rougeL'].mid.precision
        
    result_dict['rouge1_recall'] = output['rouge1'].mid.recall
    result_dict['rouge2_recall'] = output['rouge2'].mid.recall
    result_dict['rougeL_recall'] = output['rougeL'].mid.recall
        
    result_dict['rouge1_fmeasure'] = output['rouge1'].mid.fmeasure
    result_dict['rouge2_fmeasure'] = output['rouge2'].mid.fmeasure
    result_dict['rougeL_fmeasure'] = output['rougeL'].mid.fmeasure
   
    #print('result_dict: ', result_dict)
    return result_dict  


def compute_bertscore_batch(predictions, references):

    result_dict = {}
    prel, re, f1 = [], [], []

    #scorer = datasets.load_metric('bertscore')
    
    for pre, ref in zip(predictions, references):
        print('pre: ', pre)
        print('ref: ', ref)
       
        output = bertscore.compute(predictions=[pre], references=[ref], lang='en')       
        prel.append(output['precision'][0])
        re.append(output['recall'][0])
        f1.append(output['f1'][0])
        print('output: ', output)
        print('-------------------')

    result_dict = {}
    result_dict['bertscore_precision'] = sum(prel)/len(prel)
    result_dict['bertscore_recall'] = sum(re)/len(re)
    result_dict['bertscore_f1'] = sum(f1)/len(f1)

    return result_dict

def compute_bleu_single(prediction, reference):

    result_dict = {}
    
    if (type(prediction) != str or type(reference) != str):
        print('"prediction" or "reference" is not a string!')
        return result_dict

    prediction = prediction.split()
    reference = reference.split()

    #scorer = datasets.load_metric('bleu')
    
    output = {}
    try:
        output = bleu.compute(predictions=[prediction], references=[[reference]])
    except:
        result_dict['bleu'] = 0
        return result_dict

    #print('output: ', output)
        
    result_dict['bleu'] = output['bleu']

    #print('result_dict: ', result_dict)
    return result_dict

def compute_bleu_batch(predictions, references):

    result_dict = {}

    if (type(predictions) != list or type(references) != list):
        print('"predictions" or "references" is not a list!')
        return result_dict

    predictions = [[x.strip('.') for x in pre.split()] for pre in predictions]
    references = [[x.strip('.') for x in ref.split()] for ref in references]

    #print('predictions: ', predictions)
    #print('references: ', [references])
    
    #scorer = datasets.load_metric('bleu')
    
    output = {}
    try:
        output = bleu.compute(predictions=predictions, references=[references])
    except:
        result_dict['bleu'] = 0
        return result_dict

    #print('output: ', output)
        
    result_dict['bleu'] = output['bleu']

    #print('result_dict: ', result_dict)
    return result_dict


def compute_meteor_single(prediction, reference):

    result_dict = {}
    
    if (type(prediction) != str or type(reference) != str):
        print('"prediction" or "reference" is not a string!')
        return result_dict

    #scorer = datasets.load_metric('meteor')
    output = meteor.compute(predictions=[prediction], references=[reference])

    #print('output: ', output)
        
    result_dict['meteor'] = output['meteor']

    #print('result_dict: ', result_dict)
    return result_dict


def compute_meteor_batch(predictions, references):

    result_dict = {}
    
    if (type(predictions) != list or type(references) != list):
        print('"predictions" or "references" is not a list!')
        return result_dict

    #scorer = datasets.load_metric('meteor')
    output = meteor.compute(predictions=[predictions], references=[references])
    #print('output: ', output)
    result_dict['meteor'] = output['meteor']

    #print('result_dict: ', result_dict)
    return result_dict


def compute_bertscore_single(prediction, reference):

    result_dict = {}
    
    if (type(prediction) != str or type(reference) != str):
        print('"prediction" or "reference" is not a string!')
        return result_dict
    
    # microsoft/deberta-xlarge-mnli (best model), model_type=roberta-large (default)
    #scorer = datasets.load_metric('bertscore')
    output = bertscore.compute(predictions=[prediction], references=[reference], lang='en')
        
    result_dict['bertscore_precision'] = output['precision']
    result_dict['bertscore_recall'] = output['recall']
    result_dict['bertscore_f1'] = output['f1']

    return result_dict  

def compute_moverscore_single(prediction, reference):

    result_dict = {}
    
    if (type(prediction) != str or type(reference) != str):
        print('"prediction" or "reference" is not a string!')
        return result_dict
    
    # Transformers==3.1.0, some errors
    # DistilBERT (original BERTMNLI)
    idf_dict_ref = get_idf_dict(reference) # idf_dict_ref = defaultdict(lambda: 1.)
    idf_dict_hyp = get_idf_dict(prediction) # idf_dict_hyp = defaultdict(lambda: 1.)

    #print('idf_dict_ref: ', idf_dict_ref)
    #print('idf_dict_hyp: ', idf_dict_hyp)

    score = word_mover_score(reference, prediction, idf_dict_ref, idf_dict_hyp, \
                             stop_words=[], n_gram=1, remove_subwords=True)
    #print('score:' , score)'''
    return result_dict

#..................

'''from nltk.translate.bleu_score import sentence_bleu

reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'sleepy', 'dog']


reference1 = ' '.join(x for x in reference[0])
candidate1 = ' '.join(x for x in candidate)

print('nltk: ', sentence_bleu(reference, candidate))
'''
'''from sacrebleu.metrics import BLEU, CHRF, TER
import sacrebleu
bleu = BLEU()


predictions = ['The', 'film', 'Death', 'on', 'a', 'Factory', 'Farm', 'was', 'shown', 'on', 'HBO', 'and', \
               'it', 'was', 'written', 'and', 'directed', 'by', 'Tom', 'Simon', 'Geof', 'Bartz', 'is', 'the', \
               'editor', 'of', 'the', 'film']
references =  [['Death', 'on', 'a', 'Factory', 'Farm', 'is', 'an', 'HBO', 'Film', 'directed', 'and', 'produced', \
                 'by', 'Tom', 'Simon', 'Geof', 'Bartz', 'is', 'the', 'editor'],
                ['Death', 'on', 'a', 'Factory', 'Farm', 'is', 'an', 'HBO', 'film', 'directed,', 'produced,', 'and', \
                 'edited', 'by', 'Geof', 'Bartz', 'and', 'Tom', 'Simon'],
                ['Tom', 'Simon', 'directed', 'and', 'produced', 'the', 'film', '‘Death', 'on', 'a', 'Factory', 'Farm’,', \
                 'which', 'was', 'broadcasted', 'by', 'HBO', 'Geof', 'Bartz', 'on', 'the', 'other', 'hand', 'edited', 'the', \
                 'film']]
                
references1 = [' '.join(x for x in reference) for reference in references]
predictions1 = [' '.join(x for x in predictions)]

print('references1: ', references1)
print('predictions1: ', predictions1)

#print('bleu: ', bleu.corpus_score(predictions1, [references1]))
print('datasets: ', compute_bleu_batch(predictions1, references1))   
print('sacrebleu: ', sacrebleu.corpus_bleu(predictions1, [references1]))

#from nltk.translate.bleu_score import sentence_bleu
#print('nltk: ', sentence_bleu(references, predictions))'''
