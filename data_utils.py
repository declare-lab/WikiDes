import torch
import random

class PostEvalDataset():
    def __init__(self, sources, candidates, targets, tokenizer, max_length):
        self.sources = sources
        self.candidates = candidates
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_cans = 5
        self.transform = int

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, item):
    
        # ignore padding in the loss, source_ids = input_ids
        #batch['candidate_ids'] = [[0 if token == tokenizer.pad_token_id else token for token in l]
        #              for l in candidate_tokenized["input_ids"]]
    
        source = str(self.sources[item])
        
        #source_tokenized = tokenizer(source, padding="max_length", truncation=True, max_length=max_source_length)
        source_tokenized = self.tokenizer.encode_plus(
                source,
                add_special_tokens=True,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_token_type_ids=False,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            

        '''candidate = ' [SEP] '.join(c for c in self.candidates[item][0:5]) # get 5 summaries
        candidate_tokenized = self.tokenizer.encode_plus(
                candidate,
                add_special_tokens=True,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_token_type_ids=False,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
            )'''

        target = str(self.targets[item])
        target_tokenized = self.tokenizer.encode_plus(
                target,
                add_special_tokens=True,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_token_type_ids=False,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

        #print('target: ', target)
        #print('target_tokenized: ', target_tokenized)
        
        candidate = self.candidates[item][:self.num_cans] 
        candidate = list(set(candidate))

        # self.candidates
        while(len(candidate) < self.num_cans): 
            #item = random.choice(self.candidates)
            #can = random.choice(item)

            can = random.choice(candidate) # repeat available candidates
            candidate.append(can) 

            '''if (can not in candidate):
                candidate.append(can)'''
        
        candidate_tokenized = []
        for can in candidate:
             item_tokenized = self.tokenizer.encode_plus(
                    can,
                    add_special_tokens=True,
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True,
                    return_token_type_ids=False,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )
             candidate_tokenized.append(item_tokenized)


        candidate_ids = [c['input_ids'].flatten() for c in candidate_tokenized]
        candidate_ids = [c.numpy() for c in candidate_ids]
        candidate_ids = torch.tensor(candidate_ids, dtype=torch.long)
        
        
        return {
            'source': source,
            'input_ids': source_tokenized['input_ids'].flatten(),
            'attention_mask': source_tokenized['attention_mask'].flatten(),
            'candidate': candidate,
            'candidate_ids': candidate_ids,
            'target': target,
            'target_ids': target_tokenized['input_ids'].flatten(),
            }
