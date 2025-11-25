#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import json

with open("Qwen2.5_VL/tracing_information_flow/datasets/chartqa/correct_answers_chartqa.json", "r") as f:
    results = json.load(f)

# %%######
def split_by_sublist(big_list, sub_list):
    n = len(sub_list)
    for i in range(len(big_list) - n + 1):
        if big_list[i:i+n] == sub_list:
            before = big_list[:i+n]  # include sub_list
            after = big_list[i+n:]   # dopo sub_list
            return before, after
    return None, None  # se sub non trovato


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Costruzione new_results veloce
new_results = []
sub1 = [19357, 21806, 25] 
count1 = 0
sub2 = [96704, 640, 25] 
count2 = 0
sub3 = [19357, 21806, 66963] 
count3 = 0

for result in results:  # 2067 results
    image_id = result['image_id']
    question = result['question']
    answer_tokens = result['answer_tokens']
    
    # Pulisci dai pad_id_tokens
    while answer_tokens and answer_tokens[-1] == 151645:
        answer_tokens = answer_tokens[:-1]
    result['answer_tokens'] = answer_tokens
    
    before, after = split_by_sublist(answer_tokens, sub1)
    if before is not None and after is not None:
        print("PRIMA (incluso sub):", before)
        print("DOPO (senza sub):", after)
        result['final_tokens'] = after
        result['thought_tokens'] = before
        new_results.append(result)
        count1 += 1
        continue

    before, after = split_by_sublist(answer_tokens, sub2)
    if before is not None and after is not None:
        print("PRIMA (incluso sub):", before)
        print("DOPO (senza sub):", after)
        result['final_tokens'] = after
        result['thought_tokens'] = before
        new_results.append(result)
        count2 += 1
        continue

    before, after = split_by_sublist(answer_tokens, sub3)
    if before is not None and after is not None:
        print("PRIMA (incluso sub):", before)
        print("DOPO (senza sub):", after)
        result['final_tokens'] = after
        result['thought_tokens'] = before
        new_results.append(result)
        count3 += 1
        continue

# %%###############
# Scrittura su file
with open("Qwen2.5_VL/tracing_information_flow/datasets/chartqa/chartqa_dataset_correct_answers.json", "w") as f:
    json.dump(new_results, f, indent=4)
# %%##############
# average length of answer tokens 
len_answ = 0
for new_result in new_results:
    len_answ += len(new_result['answer_tokens'])

print(f"{len_answ/len(new_results)} average length of answer tokens")

# average length of thought tokens 
len_th = 0
for new_result in new_results:
    len_th += len(new_result['thought_tokens'])

print(f"{len_th/len(new_results)} average length of thought tokens")
