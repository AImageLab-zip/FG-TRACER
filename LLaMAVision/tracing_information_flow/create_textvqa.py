#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import json

with open("LLaMAVision/tracing_information_flow/dataset/textvqa/correct_answers_textvqa.json", "r") as f:
    results = json.load(f)
    
with open("/path/to/TextVQA/TextVQA_0.5.1_val.json", "r") as fp:
    data = json.load(fp)

with open("LLaMAVision/tracing_information_flow/dataset/textvqa/filter_image_classes_textvqa.json", "r") as fp:
    filter_image_classes = json.load(fp)

datas = data['data'] #5000 samples
print(len(results))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correct answer file
# Pre-processing: creiamo lookup table per accesso rapido
datas_dict = {sample['question_id']: sample for sample in datas}
filter_classes = list(filter_image_classes.keys())
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Costruzione new_results veloce
new_results = []
for result in results:  # 3926 results -> 2680 samples in filtered_classes
    question_id = result['questionId']
    sample = datas_dict[question_id]
    for img_class in sample['image_classes']:
        if img_class in filter_classes:
            answer_tokens = result['answer_tokens']
            # Pulisci dai pad_id_tokens
            while answer_tokens and answer_tokens[-1] == 128004:
                answer_tokens = answer_tokens[:-1]
            result['answer_tokens'] = answer_tokens 
            result['image_classes'] = sample['image_classes']
            result['image_id'] = sample['image_id']
    
            new_results.append(result)
            break
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
with open("LLaMAVision/tracing_information_flow/dataset/textvqa/textvqa_dataset_correct_answers.json", "w") as f:
    json.dump(new_results, f, indent=4)
# %%
# correct answers' ids file  
id_q_types = {}
for result in new_results:
    for q_type in result['image_classes']:
        if q_type in filter_classes:
            if q_type in id_q_types:
                id_q_types[q_type].append(result['questionId'])
            else:
                id_q_types[q_type] = [result['questionId']]
# %% 
with open("LLaMAVision/tracing_information_flow/dataset/textvqa/ids_correct_textvqa.json", "w") as f:
    json.dump(id_q_types, f, indent=2)
# %% 
question_types_tot = {}
for sample in datas:
    for q_type in sample['image_classes']:
        if q_type in filter_classes:
            if q_type in question_types_tot:
                question_types_tot[q_type].append(sample['question_id'])
            else:
                question_types_tot[q_type] = [sample['question_id']]
# %%           
# statistics for each question type
for q_type in filter_classes:
    print(f"{q_type} correct: {len(id_q_types[q_type])}")
    print(f"{q_type} total: {len(question_types_tot[q_type])}")
    print(f"{q_type} percentage: {len(id_q_types[q_type])/len(question_types_tot[q_type])}")
# %%
# average length of answer tokens for chosen image category 
for category in filter_classes:
    len_answ = 0
    for new_result in new_results:
        if category in new_result['image_classes']:
            len_answ += len(new_result['answer_tokens'])

    print(f"{category}: {len_answ/len(id_q_types[category])}")


# %%
