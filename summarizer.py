import torch
import numpy as np
import datasets

import sys
sys.setrecursionlimit(10**6)

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from read_write_file import *

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)

from metrics import *

class Config:
    def __init__(self, model = 'facebook/bart-base', tokenizer = 'facebook/bart-base', batch_size = 16, \
                  encoder_max_length = 256, decoder_max_length = 16, num_train_epochs = 3):
        super(Config, self).__init__()

        self.seed = 42
        self.model_name = model.replace('/','_')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model, do_lower_case = False)
    
        self.batch_size = batch_size
        self.encoder_max_length = encoder_max_length
        self.decoder_max_length = decoder_max_length

        self.num_train_epochs = num_train_epochs

def batch_tokenize_preprocess(batch, tokenizer, max_source_length, max_target_length):
    source, target = batch["source"], batch["target"]
    source_tokenized = tokenizer(source, padding="max_length", truncation=True, max_length=max_source_length)
    batch = {k: v for k, v in source_tokenized.items()}

    target_tokenized = tokenizer(target, padding="max_length", truncation=True, max_length=max_target_length)

    # Ignore padding in the loss
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in l]
                       for l in target_tokenized["input_ids"]]
    #batch["labels"] = [[token for token in l] for l in target_tokenized["input_ids"]]
    #print('batch: ', len(batch))
                       
    return batch

def load_data(batch_size, tokenizer, encoder_max_length, decoder_max_length, train_file = 'dataset/phrase1/training_para_256.json', \
              val_file = 'dataset/phrase1/validation_para_256.json'):
    
    train_data = datasets.load_dataset('json', data_files = train_file)
    val_data = datasets.load_dataset('json', data_files = val_file)

    train_data = train_data.map(lambda batch: batch_tokenize_preprocess(
            batch, tokenizer, encoder_max_length, decoder_max_length),
            batched=True,
            batch_size=batch_size,
            remove_columns=['source', 'target', 'label', 'wikidata_id', 'baseline_candidates']
        )

    
    val_data = val_data.map(lambda batch: batch_tokenize_preprocess(
            batch, tokenizer, encoder_max_length, decoder_max_length),
            batched=True,
            batch_size=batch_size,
            remove_columns=['source', 'target', 'label', 'wikidata_id', 'baseline_candidates']
        )

    # there are an error with tokenizer on datasets
    '''for item in train_data['train']:
        print('item: ', item)
        print('-------------------')'''

    #train_data = train_data['train'].remove_columns('baseline_candidates')
    #val_data = val_data['train'].remove_columns('baseline_candidates')
    #tokenized_datasets = tokenized_datasets.remove_columns_(books_dataset["train"].column_names)
    print('train_data: ', len(train_data['train']))
    print('val_data: ', len(val_data['train']))

    return train_data, val_data


def compute_metrics(pred, metrics = ['rouge']):

    # bertscore, bleu, bleurt, coval, gleu, glue, meteor, \
    # rouge, sacrebleu, seqeval, squad, squad_v2, xnli\
    
    # bleu has error, https://github.com/huggingface/evaluate/issues/107

    # load metrics
    metric_list = []
    for m in metrics: 
        metric_list.append([m, datasets.load_metric(m)])
    
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
    print('pred_str: ', pred_str[0])
    print('label_str: ', label_str[0])

    metric_outputs = []
    for m in metric_list:
        if (m[0] == 'bleu'):
            metric_outputs.append({'metric': m[0], 'compute': m[1].compute(predictions=[pred_str], references=[label_str])})
        elif (m[0] == 'rouge'):
            metric_outputs.append({'metric': m[0], 'compute': m[1].compute(predictions=pred_str, references=label_str)})
        elif (m[0] == 'bertscore'):
            metric_outputs.append({'metric': m[0], 'compute': m[1].compute(predictions=pred_str, references=label_str, lang='en')})
   
    result_dict = {}
    for output in metric_outputs:
        if (output['metric'] == 'rouge'):
            result_dict['rouge1_precision'] = output['compute']['rouge1'].mid.precision
            result_dict['rouge2_precision'] = output['compute']['rouge2'].mid.precision
            result_dict['rougeL_precision'] = output['compute']['rougeL'].mid.precision
            
            result_dict['rouge1_recall'] = output['compute']['rouge1'].mid.recall
            result_dict['rouge2_recall'] = output['compute']['rouge2'].mid.recall
            result_dict['rougeL_recall'] = output['compute']['rougeL'].mid.recall
            
            result_dict['rouge1_fmeasure'] = output['compute']['rouge1'].mid.fmeasure
            result_dict['rouge2_fmeasure'] = output['compute']['rouge2'].mid.fmeasure
            result_dict['rougeL_fmeasure'] = output['compute']['rougeL'].mid.fmeasure
        '''if (output['metric'] == 'bertscore'): # out of memory
            print('output: ', output)
            result_dict['bertscore_precision'] = output['compute']['precision']
            result_dict['bertscore_recall'] = output['compute']['recall']
            result_dict['bertscore_f1'] = output['compute']['f1']'''
       
    return result_dict
    


def train(model, tokenizer, train_data, val_data, num_train_epochs = 5, batch_size = 16, output_dir = 'output/phrase1'):
    
    save_steps = len(train_data['train'])//batch_size
    warmup_steps = save_steps//10 # 10% warmup

    training_args = Seq2SeqTrainingArguments(

        #weight_decay=0.1,
        #label_smoothing_factor=0.1,
        #logging_dir="logs",
        #learning_rate=3e-05,
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        #evaluate_during_training=True,
        evaluation_strategy='epoch',
        do_train=True,
        do_eval=True,
        logging_steps = save_steps,  
        #save_steps = save_steps, 
        #eval_steps = save_steps,  
        warmup_steps = warmup_steps,  
        #max_steps = 16,
        overwrite_output_dir = True,
        save_total_limit = 2,
        #save_strategy = "no",
        #load_best_model_at_end = False,
        num_train_epochs = num_train_epochs,
        fp16 = True, 
        metric_for_best_model = 'eval_loss',
        load_best_model_at_end = True # will ignore save steps, saved after each evaluation
    )

    # data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        #data_collator=data_collator,
        train_dataset=train_data['train'],
        eval_dataset=val_data['train'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
        #callbacks=[CustomCallback]
    )

    trainer.train()

'''class CustomCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def on_epoch_begin(self, args, state, control, **kwargs):
    
        print("Starting training")
        
        # get hidden states
        
        if torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)'''

def generate_single(model, tokenizer, text, max_length = 256, num_beams = 1):

    # num_beams = 1 means no beam search
    # https://huggingface.co/docs/transformers/v4.19.4/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.bos_token_id

    # tokenizer will automatically set [BOS] <text> [EOS] cut off at max length
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    # num_beams + max_length
    
    if (num_beams < 1):
        outputs = model.generate(input_ids, attention_mask = attention_mask, max_length = max_length)
    else:
        outputs = model.generate(input_ids, attention_mask = attention_mask, num_beams = num_beams, max_length = max_length)

    # all special tokens including will be removed
    pred_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #print('pred_text: ', pred_text)
    return pred_text[0]

def generate_data(model_url = 'output/checkpoint', tokenizer_name = 'facebook/bart-base', batch_size = 64, \
                  test_file = 'dataset/validation_data.csv'):
    

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case = False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_url)

    model.to("cuda")
    model.eval()

    test_data = datasets.load_dataset('csv', data_files = test_file)
    #test_data = test_data.select(range(16))

    results = test_data.map(generate_summary, batched=True)
    print(results)

    '''label_str = results['train']['target']
    pred_str = results['train']['pred']

    rouge_output1 = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge1"])["rouge1"].mid
    rouge_output2 = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    print('rouge 1: ', rouge_output1)
    print('rouge 2: ', rouge_output2)'''


def collect_single_summary(item, model, tokenizer, max_num_beams = 25, output_file = ''):

    candidate_list = []
    print(item['source'])
    
    for n in range(1, max_num_beams + 1):
        pred_text = generate_single(model, tokenizer, item['source'], max_length = 256, num_beams = n)
        if (pred_text.strip() not in candidate_list):
            candidate_list.append(pred_text.strip())

    for c in item['baseline_candidates']:
        if (c not in candidate_list):
            candidate_list.append(c)
                
    # add target?
    #candidate_list.insert(0, item['target'])
        
    # filter repetition
    candidate_list = list(set(candidate_list))

    item_dict = {'source': item['source'], 'candidate':candidate_list, 'target': item['target']}
    write_single_dict_to_jsonl_file(output_file, item_dict)
    
    print('------------------------------------')
    

def collect_summary(model, tokenizer, max_num_beams = 25, max_workers = 4, input_file = 'dataset/phrase1/training_para_64.json', \
                    output_file = ''):
  
    """
        collect training data for the PostEval model
    """
    
    data = datasets.load_dataset('json', data_files = input_file)
    
    size = len(data['train'])
    print('size: ', size)

    with ThreadPoolExecutor(max_workers = max_workers) as executor:
        results = executor.map(collect_single_summary, data['train'], [model]*size, [tokenizer]*size, [max_num_beams]*size, \
                               [output_file]*size, timeout = 600)
    
    '''for item in data['train']:
        collect_single_summary(item, model, tokenizer, max_num_beams = 50, output_file = 'dataset/collect_sum.json')'''

def eval_dataset_baseline_bertscore(input_file = 'dataset/phrase1/test_para_256.json', output_dir='output'):

    dataset = load_list_from_jsonl_file(input_file)
    
    bert_f1 = []

    references = [item['target'] for item in dataset]
    predictions = []
    for item in dataset:
        baseline = ''
        try: baseline = item['baseline_candidates'][0]
        except: pass
        predictions.append(baseline)
    bert_ouput = compute_bertscore_batch(predictions, references)
    
    result_dict = {}
    result_dict['file'] = input_file
    result_dict['model_name'] = 'baseline'
    result_dict['bertscore_f1'] = bert_ouput['bertscore_f1']
    result_dict['bertscore_precision'] = bert_ouput['bertscore_precision']
    result_dict['bertscore_recall'] = bert_ouput['bertscore_recall']

    write_single_dict_to_json_file(output_dir + '/result.json', result_dict, file_access = 'a')
    return result_dict

def eval_dataset_baseline(input_file = 'dataset/phrase1/test_para_256.json', output_dir='output'):

    dataset = load_list_from_jsonl_file(input_file)
    
    r1_pre, r2_pre, rL_pre = [], [], []
    r1_re, r2_re, rL_re = [], [], []
    r1_fm, r2_fm, rL_fm = [], [], []
    
    bleu = []
    meteor = []

    i = 0
    for item in dataset:

        print('Checking item: ', input_file, str(i) + '/' + str(len(dataset)))
        i = i + 1

        #prediction = generate_single(model, tokenizer, item['source'], max_length = 256, num_beams = 0)
        target = item['target']

        # extract only the first item
        baseline = ''
        try:
            baseline = item['baseline_candidates'][0]
        except:
            pass

        if (baseline == ''):
            r1_pre.append(0)
            r2_pre.append(0)
            rL_pre.append(0)

            r1_re.append(0)
            r2_re.append(0)
            rL_re.append(0)

            r1_fm.append(0)
            r2_fm.append(0)
            rL_fm.append(0)
            continue
    
        output = compute_rouge_single(baseline, target)

        r1_pre.append(output['rouge1_precision'])
        r2_pre.append(output['rouge2_precision'])
        rL_pre.append(output['rougeL_precision'])

        r1_re.append(output['rouge1_recall'])
        r2_re.append(output['rouge2_recall'])
        rL_re.append(output['rougeL_recall'])

        r1_fm.append(output['rouge1_fmeasure'])
        r2_fm.append(output['rouge2_fmeasure'])
        rL_fm.append(output['rougeL_fmeasure'])

        print('baseline: ', baseline)
        print('target: ', target)

        output = compute_bleu_single(baseline, target)
        bleu.append(output['bleu'])

        output = compute_meteor_single(baseline, target)
        meteor.append(output['meteor'])

        print('----------------')
    
    r1_pre = sum(r1_pre)/len(r1_pre)
    r2_pre = sum(r2_pre)/len(r2_pre)
    rL_pre = sum(rL_pre)/len(rL_pre)
    
    r1_re = sum(r1_re)/len(r1_re)
    r2_re = sum(r2_re)/len(r2_re)
    rL_re = sum(rL_re)/len(rL_re)
    
    r1_fm = sum(r1_fm)/len(r1_fm)
    r2_fm = sum(r2_fm)/len(r2_fm)
    rL_fm = sum(rL_fm)/len(rL_fm)

    
    bleu = sum(bleu)/len(bleu)
    meteor = sum(meteor)/len(meteor)

    result_dict = {}
    result_dict['file'] = input_file
    result_dict['model_name'] = 'baseline'
    result_dict['rouge1_precision'] = r1_pre
    result_dict['rouge2_precision'] = r2_pre
    result_dict['rougeL_precision'] = rL_pre
        
    result_dict['rouge1_recall'] = r1_re
    result_dict['rouge2_recall'] = r2_re
    result_dict['rougeL_recall'] = rL_re
        
    result_dict['rouge1_fmeasure'] = r1_fm
    result_dict['rouge2_fmeasure'] = r2_fm
    result_dict['rougeL_fmeasure'] = rL_fm

    result_dict['bleu'] = bleu
    result_dict['meteor'] = meteor
    

    write_single_dict_to_json_file(output_dir + '/result.json', result_dict, file_access = 'a')
    return result_dict


def eval_dataset_bertscore(model_name, model, tokenizer, input_file = 'dataset/phrase1/test_para_256.json', output_dir='output'):

    dataset = load_list_from_jsonl_file(input_file)
    
    bert_f1 = []

    predictions = []
    references = []
    
    for item in dataset:
    
        references.append(item['target'])

        prediction = generate_single(model, tokenizer, item['source'], max_length = 256, num_beams = 0)
        predictions.append(prediction)   
        
    bert_ouput = compute_bertscore_batch(predictions, references)
    
    result_dict = {}
    result_dict['file'] = input_file
    result_dict['model_name'] = model_name
    result_dict['bertscore_f1'] = bert_ouput['bertscore_f1']
    result_dict['bertscore_precision'] = bert_ouput['bertscore_precision']
    result_dict['bertscore_recall'] = bert_ouput['bertscore_recall']
    
    write_single_dict_to_json_file(output_dir + '/result.json', result_dict, file_access = 'a')
    return result_dict

def eval_dataset(model_name, model, tokenizer, input_file = 'dataset/phrase1/test_para_256.json', output_dir='output'):

    dataset = load_list_from_jsonl_file(input_file)
    
    r1_pre, r2_pre, rL_pre = [], [], []
    r1_re, r2_re, rL_re = [], [], []
    r1_fm, r2_fm, rL_fm = [], [], []


    #bert_f1 = []
    bleu = []
    meteor = []
    
    i = 0
    for item in dataset:

        print('Checking item: ', model_name, str(i) + '/' + str(len(dataset)))
        i = i + 1

        prediction = generate_single(model, tokenizer, item['source'], max_length = 256, num_beams = 0)
        target = item['target']
    
        '''output = compute_rouge_single(prediction, target)

        r1_pre.append(output['rouge1_precision'])
        r2_pre.append(output['rouge2_precision'])
        rL_pre.append(output['rougeL_precision'])

        r1_re.append(output['rouge1_recall'])
        r2_re.append(output['rouge2_recall'])
        rL_re.append(output['rougeL_recall'])

        r1_fm.append(output['rouge1_fmeasure'])
        r2_fm.append(output['rouge2_fmeasure'])
        rL_fm.append(output['rougeL_fmeasure'])'''

        #output = compute_bertscore_single(prediction, target)
        #bert_f1.append(output['bertscore_f1'])
            
        output = compute_bleu_single(prediction, target)
        bleu.append(output['bleu'])

        output = compute_meteor_single(prediction, target)
        meteor.append(output['meteor'])

    
    '''r1_pre = sum(r1_pre)/len(r1_pre)
    r2_pre = sum(r2_pre)/len(r2_pre)
    rL_pre = sum(rL_pre)/len(rL_pre)
    
    r1_re = sum(r1_re)/len(r1_re)
    r2_re = sum(r2_re)/len(r2_re)
    rL_re = sum(rL_re)/len(rL_re)
    
    r1_fm = sum(r1_fm)/len(r1_fm)
    r2_fm = sum(r2_fm)/len(r2_fm)
    rL_fm = sum(rL_fm)/len(rL_fm)'''

    #bert_f1 = sum(bert_f1)/len(bert_f1)
    bleu = sum(bleu)/len(bleu)
    meteor = sum(meteor)/len(meteor)

    result_dict = {}
    result_dict['file'] = input_file
    result_dict['model_name'] = model_name
    '''result_dict['rouge1_precision'] = r1_pre
    result_dict['rouge2_precision'] = r2_pre
    result_dict['rougeL_precision'] = rL_pre
        
    result_dict['rouge1_recall'] = r1_re
    result_dict['rouge2_recall'] = r2_re
    result_dict['rougeL_recall'] = rL_re
        
    result_dict['rouge1_fmeasure'] = r1_fm
    result_dict['rouge2_fmeasure'] = r2_fm
    result_dict['rougeL_fmeasure'] = rL_fm'''

    #result_dict['bert_f1'] = bert_f1
    result_dict['bleu'] = bleu
    result_dict['meteor'] = meteor

    write_single_dict_to_json_file(output_dir + '/result.json', result_dict, file_access = 'a')
    return result_dict

def eval_dataset_bertscore_source(model_name, model, tokenizer, input_file = 'dataset/phrase1/test_para_256.json', output_dir='output'):

    dataset = load_list_from_jsonl_file(input_file)
    
    bert_f1 = []

    predictions = []
    references = []
    
    for item in dataset:
    
        references.append(item['source'])

        prediction = generate_single(model, tokenizer, item['source'], max_length = 256, num_beams = 0)
        predictions.append(prediction)   
        
    bert_ouput = compute_bertscore_batch(predictions, references)
    
    result_dict = {}
    result_dict['file'] = input_file
    result_dict['model_name'] = model_name
    result_dict['bertscore_f1'] = bert_ouput['bertscore_f1']
    result_dict['bertscore_precision'] = bert_ouput['bertscore_precision']
    result_dict['bertscore_recall'] = bert_ouput['bertscore_recall']
    
    write_single_dict_to_json_file(output_dir + '/result.json', result_dict, file_access = 'a')
    return result_dict

def eval_dataset_source(model_name, model, tokenizer, input_file = 'dataset/phrase1/test_para_256.json', output_dir='output'):

    dataset = load_list_from_jsonl_file(input_file)
    
    r1_pre, r2_pre, rL_pre = [], [], []
    r1_re, r2_re, rL_re = [], [], []
    r1_fm, r2_fm, rL_fm = [], [], []

    #bert_f1 = []
    bleu = []
    meteor = []

    i = 0
    for item in dataset:

        print('Checking item: ', model_name, str(i) + '/' + str(len(dataset)))
        i = i + 1

        prediction = generate_single(model, tokenizer, item['source'], max_length = 256, num_beams = 0)
        output = compute_rouge_single(prediction, item['source'])

        r1_pre.append(output['rouge1_precision'])
        r2_pre.append(output['rouge2_precision'])
        rL_pre.append(output['rougeL_precision'])

        r1_re.append(output['rouge1_recall'])
        r2_re.append(output['rouge2_recall'])
        rL_re.append(output['rougeL_recall'])

        r1_fm.append(output['rouge1_fmeasure'])
        r2_fm.append(output['rouge2_fmeasure'])
        rL_fm.append(output['rougeL_fmeasure'])

        #output = compute_bertscore_single(prediction, item['source'])
        #bert_f1.append(output['bertscore_f1'])

        output = compute_bleu_single(prediction, item['source'])
        bleu.append(output['bleu'])

        output = compute_meteor_single(prediction, item['source'])
        meteor.append(output['meteor'])

    
    r1_pre = sum(r1_pre)/len(r1_pre)
    r2_pre = sum(r2_pre)/len(r2_pre)
    rL_pre = sum(rL_pre)/len(rL_pre)
    
    r1_re = sum(r1_re)/len(r1_re)
    r2_re = sum(r2_re)/len(r2_re)
    rL_re = sum(rL_re)/len(rL_re)
    
    r1_fm = sum(r1_fm)/len(r1_fm)
    r2_fm = sum(r2_fm)/len(r2_fm)
    rL_fm = sum(rL_fm)/len(rL_fm)

    #bert_f1 = sum(bert_f1)/len(bert_f1)
    bleu = sum(bleu)/len(bleu)
    meteor = sum(meteor)/len(meteor)

    result_dict = {}
    result_dict['file'] = input_file
    result_dict['model_name'] = model_name + '_source'
    result_dict['rouge1_precision'] = r1_pre
    result_dict['rouge2_precision'] = r2_pre
    result_dict['rougeL_precision'] = rL_pre
        
    result_dict['rouge1_recall'] = r1_re
    result_dict['rouge2_recall'] = r2_re
    result_dict['rougeL_recall'] = rL_re
        
    result_dict['rouge1_fmeasure'] = r1_fm
    result_dict['rouge2_fmeasure'] = r2_fm
    result_dict['rougeL_fmeasure'] = rL_fm

    #result_dict['bert_f1'] = bert_f1
    result_dict['bleu'] = bleu
    result_dict['meteor'] = meteor

    write_single_dict_to_json_file(output_dir + '/result.json', result_dict, file_access = 'a')
    return result_dict
     
#....................................................................
# set global tokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/ssr-base', do_lower_case=False) # used for compute_metrics() function
    
if __name__ == "__main__":

    # training models: facebook/bart-base, t5-small, t5-base, microsoft/ssr-base, google/t5-v1_1-small, google/t5-v1_1-base
    '''config = Config(model = 'microsoft/ssr-base', tokenizer = 'microsoft/ssr-base', batch_size = 8, \
                    encoder_max_length = 256, decoder_max_length = 32, num_train_epochs = 3)

    train_data, val_data = load_data(config.batch_size, config.tokenizer, config.encoder_max_length, \
                                     config.decoder_max_length, train_file = 'dataset/phrase1/random/training_para_256.json', \
                                     val_file = 'dataset/phrase1/random/validation_para_256.json')

    train(config.model, config.tokenizer, train_data, val_data, num_train_epochs = config.num_train_epochs, \
          batch_size = config.batch_size, output_dir='output/' + config.model_name)'''

    # eval baselines ------------------
    '''eval_dataset_baseline(input_file = 'dataset/phrase1/diff/test_para_256.json', output_dir='output/phrase1/')
    eval_dataset_baseline(input_file = 'dataset/phrase1/diff/validation_para_256.json', output_dir='output/phrase1/')

    eval_dataset_baseline(input_file = 'dataset/phrase1/random/test_para_256.json', output_dir='output/phrase1/')
    eval_dataset_baseline(input_file = 'dataset/phrase1/random/validation_para_256.json', output_dir='output/phrase1/')'''

    '''eval_dataset_baseline_bertscore(input_file = 'dataset/phrase1/diff/test_para_256.json', output_dir='output/phrase1/')
    eval_dataset_baseline_bertscore(input_file = 'dataset/phrase1/diff/validation_para_256.json', output_dir='output/phrase1/')

    eval_dataset_baseline_bertscore(input_file = 'dataset/phrase1/random/test_para_256.json', output_dir='output/phrase1/')
    eval_dataset_baseline_bertscore(input_file = 'dataset/phrase1/random/validation_para_256.json', output_dir='output/phrase1/')'''
    
    # ---------------------------------
    
    # different topics
    # ---------------------------------
    '''model_name = 'facebook/bart-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case = False)
    model = AutoModelForSeq2SeqLM.from_pretrained('output/facebook_bart-base_diff/checkpoint-24666')

    model.to("cuda")
    model.eval()'''
    
    '''eval_dataset(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/diff/test_para_256.json', output_dir='output/phrase1/')
    eval_dataset(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/diff/validation_para_256.json', output_dir='output/phrase1/')'''
                 
    '''eval_dataset_bertscore(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/diff/test_para_256.json', output_dir='output/phrase1/')
    eval_dataset_bertscore(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/diff/validation_para_256.json', output_dir='output/phrase1/')  '''         

    
    '''model_name = 't5-small'
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case = False)
    model = AutoModelForSeq2SeqLM.from_pretrained('output/t5-small_diff/checkpoint-24666')

    model.to("cuda")
    model.eval()'''
    
    '''eval_dataset(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/diff/test_para_256.json', output_dir='output/phrase1/')
    eval_dataset(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/diff/validation_para_256.json', output_dir='output/phrase1/')'''

    '''eval_dataset_bertscore(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/diff/test_para_256.json', output_dir='output/phrase1/')
    eval_dataset_bertscore(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/diff/validation_para_256.json', output_dir='output/phrase1/')'''
    
    '''model_name = 't5-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case = False)
    model = AutoModelForSeq2SeqLM.from_pretrained('output/t5-base_diff/checkpoint-24666')

    model.to("cuda")
    model.eval()'''
    
    '''eval_dataset(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/diff/test_para_256.json', output_dir='output/phrase1/')
    eval_dataset(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/diff/validation_para_256.json', output_dir='output/phrase1/')'''
                 
    '''eval_dataset_bertscore(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/diff/test_para_256.json', output_dir='output/phrase1/')
    eval_dataset_bertscore(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/diff/validation_para_256.json', output_dir='output/phrase1/')'''     

    

    '''model_name = 'microsoft/ssr-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case = False)
    model = AutoModelForSeq2SeqLM.from_pretrained('output/microsoft_ssr-base_diff/checkpoint-24666')

    model.to("cuda")
    model.eval()
    
    eval_dataset(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/diff/test_para_256.json', output_dir='output/phrase1/')
    eval_dataset(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/diff/validation_para_256.json', output_dir='output/phrase1/')'''
                 
    '''eval_dataset_bertscore(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/diff/test_para_256.json', output_dir='output/phrase1/')
    eval_dataset_bertscore(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/diff/validation_para_256.json', output_dir='output/phrase1/')  '''
                 
    # ---------------------------------
    
    # random topics
    # ---------------------------------
    '''model_name = 'facebook/bart-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case = False)
    model = AutoModelForSeq2SeqLM.from_pretrained('output/facebook_bart-base_random/checkpoint-25611')

    model.to("cuda")
    model.eval()
    
    eval_dataset(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/random/test_para_256.json', output_dir='output/phrase1/')
    eval_dataset(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/random/validation_para_256.json', output_dir='output/phrase1/')'''

    
    '''eval_dataset_bertscore(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/random/test_para_256.json', output_dir='output/phrase1/')
    eval_dataset_bertscore(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/random/validation_para_256.json', output_dir='output/phrase1/')'''
    
    '''model_name = 't5-small'
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case = False)
    model = AutoModelForSeq2SeqLM.from_pretrained('output/t5-small_random/checkpoint-25611')

    model.to("cuda")
    model.eval()
    
    eval_dataset(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/random/test_para_256.json', output_dir='output/phrase1/')
    eval_dataset(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/random/validation_para_256.json', output_dir='output/phrase1/')'''
    
    '''eval_dataset_bertscore(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/random/test_para_256.json', output_dir='output/phrase1/')
    eval_dataset_bertscore(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/random/validation_para_256.json', output_dir='output/phrase1/')'''

    
    '''model_name = 't5-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case = False)
    model = AutoModelForSeq2SeqLM.from_pretrained('output/t5-base_random/checkpoint-25611')

    model.to("cuda")
    model.eval()
    
    eval_dataset(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/random/test_para_256.json', output_dir='output/phrase1/')
    eval_dataset(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/random/validation_para_256.json', output_dir='output/phrase1/')'''
                 
    '''eval_dataset_bertscore(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/random/test_para_256.json', output_dir='output/phrase1/')
    eval_dataset_bertscore(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/random/validation_para_256.json', output_dir='output/phrase1/')'''


    '''model_name = 'microsoft/ssr-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case = False)
    model = AutoModelForSeq2SeqLM.from_pretrained('output/microsoft_ssr-base_random/checkpoint-25611')

    model.to("cuda")
    model.eval()
    
    eval_dataset(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/random/test_para_256.json', output_dir='output/phrase1/')
    eval_dataset(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/random/validation_para_256.json', output_dir='output/phrase1/')'''
    
    '''eval_dataset_bertscore(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/random/test_para_256.json', output_dir='output/phrase1/')
    eval_dataset_bertscore(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase1/random/validation_para_256.json', output_dir='output/phrase1/')'''
    # ---------------------------------
    
    # collect summaries ---------------
    
    '''model_name = 't5-small'
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case = False)
    model = AutoModelForSeq2SeqLM.from_pretrained('output/t5-small_diff/checkpoint-24666')

    model.to("cuda")
    model.eval()'''
    
    '''collect_summary(model, tokenizer, input_file = 'dataset/phrase1/diff/training_para_256.json', \
                    output_file = 'dataset/phrase2/generated_training_para_256_diff.json')'''
                    
    '''collect_summary(model, tokenizer, input_file = 'dataset/phrase1/diff/validation_para_256.json', \
                    output_file = 'dataset/phrase2/generated_validation_para_256_diff.json')'''
    
    '''collect_summary(model, tokenizer, input_file = 'dataset/phrase1/diff/test_para_256.json', \
                    output_file = 'dataset/phrase2/generated_test_para_256_diff.json')'''
                    
    '''model_name = 'facebook/bart-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case = False)
    model = AutoModelForSeq2SeqLM.from_pretrained('output/facebook_bart-base_random/checkpoint-25611')

    model.to("cuda")
    model.eval()'''
    
    '''collect_summary(model, tokenizer, input_file = 'dataset/phrase1/random/test_para_256.json', \
                    output_file = 'dataset/phrase2/generated_test_para_256_random.json')'''
                    
    '''collect_summary(model, tokenizer, input_file = 'dataset/phrase1/random/validation_para_256.json', \
                    output_file = 'dataset/phrase2/generated_validation_para_256_random.json')'''
    
    '''collect_summary(model, tokenizer, input_file = 'dataset/phrase1/random/training_para_256.json', \
                    output_file = 'dataset/phrase2/generated_training_para_256_random.json')'''
    
    
    # Subsets of phase II -------------
    '''model_name = 't5-small'
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case = False)
    model = AutoModelForSeq2SeqLM.from_pretrained('output/t5-small_diff/checkpoint-24666')

    model.to("cuda")
    model.eval()

    eval_dataset_bertscore_source(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase2/generated_test_para_256_diff.json', output_dir='output/phrase2/')
    eval_dataset_bertscore_source(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase2/generated_validation_para_256_diff.json', output_dir='output/phrase2/')'''
    
    '''eval_dataset_source(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase2/generated_test_para_256_diff.json', output_dir='output/phrase2/')
    eval_dataset_source(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase2/generated_validation_para_256_diff.json', output_dir='output/phrase2/')'''
                 
                 
    model_name = 'facebook/bart-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case = False)
    model = AutoModelForSeq2SeqLM.from_pretrained('output/facebook_bart-base_random/checkpoint-25611')

    model.to("cuda")
    model.eval()

    '''eval_dataset_bertscore_source(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase2/generated_test_para_256_random.json', output_dir='output/phrase2/')'''
    eval_dataset_bertscore_source(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase2/generated_validation_para_256_random.json', output_dir='output/phrase2/')
    
    '''eval_dataset_source(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase2/generated_test_para_256_random.json', output_dir='output/phrase2/')
    eval_dataset_source(model_name, model, tokenizer, \
                 input_file = 'dataset/phrase2/generated_validation_para_256_random.json', output_dir='output/phrase2/')'''
    # ---------------------------------
