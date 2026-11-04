#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import json

with open("./datasets/chartqa/correct_answers_chartqa.json", "r") as f:
    results = json.load(f)

#%%%#####################################################
# CHARTQA COT COMPLETE
new_results_complete = []

for result in results:  # 2067 results
    image_id = result['image_id']
    question = result['question']
    answer_tokens = result['answer_tokens']

    # strip the leading BOS (1) and any trailing EOS (2): generation.py records neither
    if answer_tokens and answer_tokens[0] == 1:
        answer_tokens = answer_tokens[1:]
    while answer_tokens and answer_tokens[-1] == 2:
        answer_tokens = answer_tokens[:-1]

    result['answer_tokens'] = answer_tokens
    new_results_complete.append(result)
# %%##########
# Scrittura su file
with open("./datasets/chartqa/chartqa_dataset_correct_answers_complete.json", "w") as f:
    json.dump(new_results_complete, f, indent=4)
# %%###########
# average length of answer tokens 
len_answ = 0
for new_result in new_results_complete:
    len_answ += len(new_result['answer_tokens'])

print(f"{len_answ/len(new_results_complete)} average length of answer tokens")
#%%%#####
# some statistics
count = 0
for new_result in new_results_complete:
    if len(new_result['answer_tokens']) > 50:
        count += 1
print(f"{count} sample >50 answer tokens")

count = 0
for new_result in new_results_complete:
    if len(new_result['answer_tokens']) <10:
        count += 1
print(f"{count} sample <10 answer tokens")
#%%%#####
# # min/max answer token length
min = 2048
max = 0
for new_result in new_results_complete:
    if len(new_result['answer_tokens']) < min:
        min = len(new_result['answer_tokens'])
    if len(new_result['answer_tokens']) > max:  
        max = len(new_result['answer_tokens'])
print(f"{max} max answer tokens")
print(f"{min} min answer tokens")
# %%
