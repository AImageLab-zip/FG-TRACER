#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import json

with open("LLaMAVision/results/eval_coco_15000.json", "r") as f:
    results = json.load(f)
    
with open("/path/to/coco/annotations/captions_val2014.json", "r") as fp:
    data = json.load(fp)['images']

with open("LLaMAVision/tracing_information_flow/dataset/coco/ids_coco2.json", "r") as fp:
    list_ids = json.load(fp)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correct answer dataset
results_dict = {result['image_id']: result for result in results}
datas_dict = {sample['id']: sample for sample in data}

new_results = [] #1095 samples
for id in list_ids:
    sample = datas_dict[id]
    result = results_dict[id]
    answer_tokens = result['answer_tokens']

    # clean from pad_id_tokens
    while answer_tokens[-1] == 128004:
        answer_tokens= answer_tokens[:-1]
        
    result['answer_tokens'] = answer_tokens
    result['file_name'] = sample['file_name']

    new_results.append(result)
# %%
with open("LLaMAVision/tracing_information_flow/dataset/coco/coco_dataset_correct_answers.json", "w") as f:
    json.dump(new_results, f, indent=2)
# %%##############
# average length of answer tokens 
len_answ = 0
for new_result in new_results:
    len_answ += len(new_result['answer_tokens'])

print(f"{len_answ/len(new_results)} average length of answer tokens")

# %%#####################
### LLaVA sampling ######
#########################
import random
random.seed(42)

id_sampled =  random.sample(list_ids, 1200)
id_sampled.sort()

new_results_sampled = [] #1095 samples

for id in id_sampled:
    sample = datas_dict[id]
    result = results_dict[id]
    tokens_answer = result['answer_tokens']
        
    result['answer_tokens'] = answer_tokens
    result['file_name'] = sample['file_name']

    new_results_sampled.append(result)
# %%##############
len_answ = 0
for new_result in new_results_sampled:
    len_answ += len(new_result['answer_tokens'])

print(f"{len_answ/len(new_results_sampled)} average length of answer tokens")
# %%##############
with open("LLaMAVision/datasets/coco/coco_dataset_correct_answers_sampled.json", "w") as f:
    json.dump(new_results_sampled, f, indent=2)

