from read_write_file import *


input_file = 'dataset\human_eval.json'
output_file = 'dataset\human_eval_output.json'
dataset = load_list_from_jsonl_file(input_file)

print('*************************')
print('*************************')
print('HUMAN EVALUATION')
print('Instructions: ')
print('---', '1. Read the paragraph and choose "summmary 1" (1) or "summary 2" (2) as the best summary for the paragraph.')
print('---', 'For example, best_summary: 1.')

print('---', '2. For the best summary you chose, rank criteria "Adequacy" (Informativeness), "Relevance" (Relatedness), "Correctness", and "Concise" (Brief) by scores:')
print('------', '1: bad, can not use')
print('------', '2: not bad but not recommend for using')
print('------', '3: fair, but need to consider')
print('------', '4: good')
print('------', '5: perfect')
print('---', 'For example, Relevance: 1, Correctness: 2, etc.')
print('*************************')
print('*************************')

index = 1
n = len(dataset)
for item in dataset:

    annot_dict = {}
    print('Item: ', str(index) + '/' + str(n))
    print('Paragraph: ', item['source'])
    print('Summmary 1: ', item['sum1'])
    print('Summmary 2: ', item['sum2'])
    print('--------------------------------------')

    best_summary = input("Choose the best summary (1 or 2): ")
    
    adequacy = input("Adequacy (1-5): ")
    relevance = input("Relevance (1-5): ")
    correctness = input("Correctness (1-5): ")
    concise = input("Consise (1-5): ")

    annot_dict['best_summary'] = best_summary
    annot_dict['adequacy'] = adequacy
    annot_dict['relevance'] = relevance
    annot_dict['correctness'] = correctness
    annot_dict['concise'] = concise
    item['coder'] = annot_dict
    
    # reset 
    del best_summary, adequacy, relevance, correctness, concise
    
    print('*************************')
    print('*************************')

    index += 1


# save results
write_list_to_jsonl_file(output_file, dataset, file_access = 'w')


    
    
