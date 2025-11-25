#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import json

with open("LLaVA/results/coco_40504.json", "r") as f:
    results = json.load(f)
    
with open("/path/to/coco/annotations/captions_val2014.json", "r") as fp:
    data = json.load(fp)['images']

with open("LLaVA/tracing_information_flow/datasets/coco/ids_coco.json", "r") as fp:
    list_ids = json.load(fp)

results_dict = {result['image_id']: result for result in results}
datas_dict = {sample['id']: sample for sample in data}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correct answer dataset
new_results = [] #1095 samples
for id in list_ids:
    sample = datas_dict[id]
    res = results_dict[id]
    answer_tokens = res['answer_tokens']

    # clean from bos_id
    if answer_tokens[0] == 1: 
        answer_tokens= answer_tokens[1:]
        
    res['answer_tokens'] = answer_tokens
    res['file_name'] = sample['file_name']

    new_results.append(res)
# %%##############
with open("LLaVA/tracing_information_flow/datasets/coco/coco_dataset_correct_answers.json", "w") as f:
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
    #tokens_answer = tokens_dict[id]
    sample = datas_dict[id]
    result_sampled = results_dict[id]
    answer_tokens = result_sampled['answer_tokens']
    
    if answer_tokens[0] == 1: 
        answer_tokens = answer_tokens[1:]
        
    result_sampled['answer_tokens'] = answer_tokens
    result_sampled['file_name'] = sample['file_name']

    new_results_sampled.append(result_sampled)
# %%##############
len_answ = 0
for new_result in new_results_sampled:
    len_answ += len(new_result['answer_tokens'])

print(f"{len_answ/len(new_results_sampled)} average length of answer tokens")

# %%##############
with open("LLaVA/tracing_information_flow/datasets/coco/coco_dataset_correct_answers_sampled.json", "w") as f:
    json.dump(new_results_sampled, f, indent=2)
# %%########################################################
##############LLAVA-LLAMA common samples##############
########################################################
with open("LLaMAVision/tracing_information_flow/dataset/coco/ids_coco2.json", "r") as f:
    ids_llama = json.load(f)

common_ids = [id for id in list_ids if id in ids_llama]
new_results_common = [] #1095 samples

for id in common_ids:
    #tokens_answer = tokens_dict[id]
    sample = datas_dict[id]
    result = results_dict[id]
    answer_tokens = result['answer_tokens']
    
    if answer_tokens[0] == 1: 
        answer_tokens= answer_tokens[1:]
        
    result['answer_tokens'] = answer_tokens
    result['file_name'] = sample['file_name']

    new_results_common.append(result)

len_answ = 0
for new_result in new_results_common:
    len_answ += len(new_result['answer_tokens'])

print(f"{len_answ/len(new_results_common)} average length of answer tokens")
# %%##############
with open("LLaVA/tracing_information_flow/datasets/coco/coco_dataset_correct_answers_common.json", "w") as f:
    json.dump(new_results_common, f, indent=2)

