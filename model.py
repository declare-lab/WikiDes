# modified from https://github.com/yixinL7/SimCLS/blob/main/model.py

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer

from metrics import *

from sklearn.preprocessing import MinMaxScaler

device = torch.device('cuda:0')

def RankingLoss(can_score, target_score, use_can_loss = True, can_margin = 0.01, gold_margin = 0.01):
    
    loss_func = torch.nn.MarginRankingLoss(0.0)
    ones = torch.ones_like(can_score)
    total_loss = loss_func(can_score, can_score, ones) # loss = 0, just to set as a tensor
    
    loss_func = torch.nn.MarginRankingLoss(gold_margin)
    target_score = target_score.unsqueeze(1).expand_as(can_score)
    ones = torch.ones_like(target_score)
    gold_loss = loss_func(can_score, target_score, ones)
    
    '''
    # alternative code for gold_loss (same result)
    gold_loss = 0
    batch_size = can_score.size(0)
    for can, target in zip(can_score, target_score): # compare by each example
        #max_can = max(can)
        #max_target = max(target)
        loss_func = torch.nn.MarginRankingLoss(gold_margin)
        ones = torch.ones_like(can)
        loss = loss_func(can, target, ones) # gold loss
        gold_loss += loss/batch_size
    print('gold_loss: ', gold_loss)
    '''
    
    can_loss = 0
    if (use_can_loss == True):
        n = can_score.size(1)
        for i in range(1, n):
            pos_can = can_score[:, :-i]
            neg_can = can_score[:, i:]

            pos_can = pos_can.contiguous().view(-1)
            neg_can = neg_can.contiguous().view(-1)
        
            ones = torch.ones_like(pos_can)
            loss_func = torch.nn.MarginRankingLoss(can_margin * i) # or (j-i)
            loss = loss_func(pos_can, neg_can, ones)
            can_loss += loss
    
    #print('gold_loss: ', gold_loss) 
    #print('can_loss: ', can_loss) 
    
    total_loss = gold_loss + can_loss
    #total_loss = total_loss.requires_grad_()
    return total_loss


class PostEvalModel(nn.Module):
    
    def __init__(self, model_name, use_sim = True, use_rouge = True, hidden_size=768):
        super(PostEvalModel, self).__init__()
    
        self.use_sim = use_sim
        self.use_rouge = use_rouge
        self.hidden_size = hidden_size

        # default if no metric is used
        if (use_sim == False and use_rouge == False): self.use_sim = True

        if ('roberta' in model_name.lower()): # roberta-base
            self.encoder = RobertaModel.from_pretrained(model_name)
            #self.tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case = False)
            self.encoder.cuda()
            #self.sep_id = 2 # '</s>' (RoBERTa)
            #self.cls_id = 0
            self.pad_id = 1
            if ('roberta-large' in model_name.lower()): self.hidden_size = 1024
        else: # bert-base-cased
            self.encoder = BertModel.from_pretrained(model_name)
            #self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case = False)
            self.encoder.cuda()
            #self.sep_id = 102 # '[SEP]' (BERT)
            #self.cls_id = 101
            self.pad_id = 0
        

    def forward(self, batch):

        candidate_sim, target_sim = 0, 0
        if (self.use_sim == True):
            source_id, candidate_id, target_id = batch['input_ids'], batch['candidate_ids'], batch['target_ids']
        
            # save to "cuda:0"
            source_id = source_id.to(device)
            candidate_id = candidate_id.to(device)
            target_id = target_id.to(device)

            # get batch_size
            batch_size = source_id.size(0)

            # get document embedding
            input_mask = source_id != self.pad_id        
            out = self.encoder(source_id, attention_mask=input_mask)[0] # last layer
            source_emb = out[:, 0, :]
            print(source_emb.size())
            assert source_emb.size() == (batch_size, self.hidden_size) # [batch_size, hidden_size]

            # get target embedding
            input_mask = target_id != self.pad_id        
            out = self.encoder(target_id, attention_mask=input_mask)[0] # last layer
            target_emb = out[:, 0, :]
            assert target_emb.size() == (batch_size, self.hidden_size) # [batch_size, hidden_size]

            # get target-source sim
            target_sim = torch.cosine_similarity(target_emb, source_emb, dim=-1) # [batch_size, candidate_num]
            #print('target_score.size(): ', target_score.size())
            #assert target_score.size() == (batch_size)

            # get candidate embedding
            candidate_num = candidate_id.size(1)
            candidate_id = candidate_id.view(-1, candidate_id.size(-1))
            input_mask = candidate_id != self.pad_id
            out = self.encoder(candidate_id, attention_mask=input_mask)[0]
            candidate_emb = out[:, 0, :].view(batch_size, candidate_num, self.hidden_size)  # [batch_size, candidate_num, hidden_size]
            assert candidate_emb.size() == (batch_size, candidate_num, self.hidden_size)

            # get candidate-source sim
            source_emb = source_emb.unsqueeze(1).expand_as(candidate_emb)
            candidate_sim = torch.cosine_similarity(candidate_emb, source_emb, dim=-1) # [batch_size, candidate_num]
            #assert candidate_score.size() == (batch_size, candidate_num)

        candidate_rouge, target_rouge = 0, 0
        if (self.use_rouge == True):
            # get rouge scores
            source, candidate, target = batch['source'], batch['candidate'], batch['target']

            # https://stackoverflow.com/questions/67390427/rouge-score-append-a-list
            # compare each candidate (more correct but take time)
            candidate = list(map(list, zip(*candidate))) # transpose 
            candidate_rouge = []        
            for can, ref in zip(candidate, source):
                can_rouge = []
                for pre in can:
                    can_rouge.append(compute_rouge_single(pre, ref)['rouge1_fmeasure'])
                    '''print('pre: ', pre)
                    print('ref: ', ref)
                    print('rouge: ', compute_rouge_single(pre, ref)['rouge1_fmeasure'])
                    print('------------------')'''
                candidate_rouge.append(can_rouge)
            candidate_rouge = torch.tensor(candidate_rouge).to(device)

            target_rouge = []            
            for tar, ref in zip(target, source):
                target_rouge.append(compute_rouge_single(tar, ref)['rouge1_fmeasure'])      
            target_rouge = torch.tensor(target_rouge).to(device)

            #print('candidate_rouge: ', candidate_rouge)
            #print('target_rouge: ', target_rouge)
            
                
            '''
            # compare each group (use an average score)
            for can in candidate:
                candidate_rouge += compute_metric_batch(can, source, metric = 'rouge')['rouge1_fmeasure']
            candidate_rouge = candidate_rouge/len(candidate)
            target_rouge = compute_metric_batch(target, source, metric = 'rouge')['rouge1_fmeasure']
            '''
            
        # combined metrics
        candidate_score, target_score = 0, 0
        if (self.use_sim == True and self.use_rouge == True):
        
            # normalize range [0, 1], rerun the experiment
            
            candidate_score = 2*candidate_sim*candidate_rouge/(candidate_sim + candidate_rouge) # F1
            target_score = 2*target_sim*target_rouge/(target_sim + target_rouge) # F1
            
            candidate_score = candidate_score.requires_grad_()
            target_score = target_score.requires_grad_()
        elif (self.use_rouge == True):
            candidate_score = candidate_rouge
            target_score = target_rouge
            candidate_score = candidate_score.requires_grad_()
            target_score = target_score.requires_grad_()
        else: # default if no metric is used
            candidate_score = candidate_sim
            target_score = target_sim           

        #print({'candidate_score': candidate_score, 'target_score': target_score})
        return {'candidate_score': candidate_score, 'target_score': target_score}
