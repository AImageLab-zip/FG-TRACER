#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import sys
sys.path.append('./')
import ast
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CHART CoT
datas_llama_chartqa = json.load(open('LLaMAVision/tracing_information_flow/dataset/chartqa/chartqa_dataset_correct_answers.json'))
list_ids_llama_chartqa = list(range(len(datas_llama_chartqa)))
answer_path_llama_chartqa = "LLaMAVision/results/chartqa_complete"
dataset = 'chart_complete'

datas_llava_chartqa = json.load(open('LLaVA/tracing_information_flow/datasets/chartqa/chartqa_dataset_correct_answers_complete.json'))
list_ids_llava_chartqa = list(range(len(datas_llava_chartqa)))
answer_path_llava_chartqa = "LLaVA/results/chartqa"

datas_qwen_chartqa = json.load(open('Qwen2.5_VL/tracing_information_flow/datasets/chartqa/chartqa_dataset_correct_answers.json'))
list_ids_qwen_chartqa = list(range(len(datas_llava_chartqa)))
answer_path_qwen_chartqa = "Qwen2.5_VL/results/chartqa"

# COCO
datas_llama_coco = json.load(open('LLaMAVision/tracing_information_flow/dataset/coco/coco_dataset_correct_answers.json'))
list_ids_llama_coco = json.load(open('LLaMAVision/tracing_information_flow/dataset/coco/ids_coco2.json'))
answer_path_llama_coco = "LLaMAVision/results/coco"
dataset = 'coco'

datas_llava_coco = json.load(open('LLaVA/tracing_information_flow/datasets/coco/coco_dataset_correct_answers_common.json'))
list_ids_llava_coco = json.load(open('LLaVA/tracing_information_flow/datasets/coco/ids_coco.json'))
answer_path_llava_coco = "LLaVA/results/coco"

datas_qwen_coco = json.load(open('Qwen2.5_VL/tracing_information_flow/datasets/coco/coco_dataset_correct_answers.json'))
list_ids_qwen_coco = json.load(open('Qwen2.5_VL/tracing_information_flow/datasets/coco/ids_coco.json'))
answer_path_qwen_coco = "Qwen2.5_VL/results/coco"

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
################################################
# Creation of the DataFrame WORD-PER-WORD  #####
################################################

import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import spacy
import string
import re

def create_df_llama(list_ids, answer_path):
    dataframe_global = {}
    if isinstance(list_ids, dict):
        ids = []
        for k in list_ids.keys():
            ids.extend(list_ids[k])
    else:
        ids = list_ids

    for id in ids: #for each sample
        path = answer_path + "/" + str(id) + ".json"
        if os.path.exists(path): 
            sample_probs = json.load(open(path))            
            subword_count = 0
            current_word = ""

            for idx, item in enumerate(sample_probs['full_attention']): #for each answer token
                #Beginning of a word
                if item[0].startswith(" ") or (item[0] == " ") or (item[0][0] in string.punctuation and len(item[0].strip())==1 and idx<len(sample_probs['full_attention'])-1 and not bool(re.search(r'\d', sample_probs['full_attention'][idx+1][0]))) or (idx==0) or ('\n' in current_word and '\n' not in item[0]): #new word
                    if current_word: # end of a previous word
                        # Normalize the probabilities
                        p2_q2l = p2_q2l.tolist()
                        p2_i2l = p2_i2l.tolist()
                        p2_i2q = p2_i2q.tolist()
                        p2_l2l = p2_l2l.tolist()

                        p1 = p1 ** (1 / subword_count)
                        p2_q2l = [p2 ** (1 / subword_count) for p2 in p2_q2l]
                        p2_i2l = [p2 ** (1 / subword_count) for p2 in p2_i2l]
                        p2_i2q = [p2 ** (1 / subword_count) for p2 in p2_i2q]
                        p2_l2l = [p2 ** (1 / subword_count) for p2 in p2_l2l]

                        # Compute information drop
                        change_prob_q2l_neg = [((p2 - p1) / p1) * 100 if (((p2 - p1) / p1) * 100) < 0 else 0 for p2 in p2_q2l]
                        change_prob_i2l_neg = [((p2 - p1) / p1) * 100 if (((p2 - p1) / p1) * 100) < 0 else 0 for p2 in p2_i2l]
                        change_prob_i2q_neg = [((p2 - p1) / p1) * 100 if (((p2 - p1) / p1) * 100) < 0 else 0 for p2 in p2_i2q]
                        change_prob_l2l_neg = [((p2 - p1) / p1) * 100 if (((p2 - p1) / p1) * 100) < 0 else 0 for p2 in p2_l2l]
                        
                        change_prob_q2l = [((p2 - p1) / p1) * 100 for p2 in p2_q2l]
                        change_prob_i2l = [((p2 - p1) / p1) * 100 for p2 in p2_i2l]
                        change_prob_i2q = [((p2 - p1) / p1) * 100 for p2 in p2_i2q]
                        change_prob_l2l = [((p2 - p1) / p1) * 100 for p2 in p2_l2l]
                            
                        # Add to the global df
                        if current_word.lower() in dataframe_global:
                            dataframe_global[current_word.lower()]['image_to_last'] = [(a+b) for a,b in zip(change_prob_i2l_neg, dataframe_global[current_word.lower()]['image_to_last'])]
                            dataframe_global[current_word.lower()]['image_to_question'] = [(a+b) for a,b in zip(change_prob_i2q_neg, dataframe_global[current_word.lower()]['image_to_question'])]
                            dataframe_global[current_word.lower()]['question_to_last'] = [(a+b) for a,b in zip(change_prob_q2l_neg, dataframe_global[current_word.lower()]['question_to_last'])]
                            dataframe_global[current_word.lower()]['last_to_last'] = [(a+b) for a,b in zip(change_prob_l2l_neg, dataframe_global[current_word.lower()]['last_to_last'])]
                            dataframe_global[current_word.lower()]['count'] += 1
                            dataframe_global[current_word.lower()]['subword_count'] = subword_count
                        else:
                            dataframe_global[current_word.lower()] = {}
                            dataframe_global[current_word.lower()]['image_to_last'] = change_prob_i2l_neg
                            dataframe_global[current_word.lower()]['image_to_question'] = change_prob_i2q_neg
                            dataframe_global[current_word.lower()]['question_to_last'] = change_prob_q2l_neg
                            dataframe_global[current_word.lower()]['last_to_last'] = change_prob_l2l_neg
                            dataframe_global[current_word.lower()]['count'] = 1
                            dataframe_global[current_word.lower()]['subword_count'] = subword_count
                    
                    # beginning of a new word
                    if item[0]== " ":
                        current_word = ""
                        subword_count = 0
                        p1 = 1
                        continue

                    current_word = item[0].lstrip()
                    p1 = item[1]
                    subword_count = 1  # <- start counting

                    p2_i2q = np.array(sample_probs['image_to_question'][idx][1])
                    p2_i2l = np.array(sample_probs['image_to_last'][idx][1])
                    p2_q2l = np.array(sample_probs['question_to_last'][idx][1])
                    p2_l2l = np.array(sample_probs['last_to_last'][idx][1])

                else: #word continue
                    current_word += item[0]
                    p1 *= item[1]
                    subword_count += 1   

                    p2_i2q = p2_i2q * np.array(sample_probs['image_to_question'][idx][1])
                    p2_i2l = p2_i2l * np.array(sample_probs['image_to_last'][idx][1])
                    p2_q2l = p2_q2l * np.array(sample_probs['question_to_last'][idx][1])
                    p2_l2l = p2_l2l * np.array(sample_probs['last_to_last'][idx][1])

        
            if current_word: # end of a previous word
                # Normalize the probabilities
                p2_q2l = p2_q2l.tolist()
                p2_i2l = p2_i2l.tolist()
                p2_i2q = p2_i2q.tolist()
                p2_l2l = p2_l2l.tolist()

                p1 = p1 ** (1 / subword_count)
                p2_q2l = [p2 ** (1 / subword_count) for p2 in p2_q2l]
                p2_i2l = [p2 ** (1 / subword_count) for p2 in p2_i2l]
                p2_i2q = [p2 ** (1 / subword_count) for p2 in p2_i2q]
                p2_l2l = [p2 ** (1 / subword_count) for p2 in p2_l2l]

                # Compute information drop
                change_prob_q2l_neg = [((p2 - p1) / p1) * 100 if (((p2 - p1) / p1) * 100) < 0 else 0 for p2 in p2_q2l]
                change_prob_i2l_neg = [((p2 - p1) / p1) * 100 if (((p2 - p1) / p1) * 100) < 0 else 0 for p2 in p2_i2l]
                change_prob_i2q_neg = [((p2 - p1) / p1) * 100 if (((p2 - p1) / p1) * 100) < 0 else 0 for p2 in p2_i2q]
                change_prob_l2l_neg = [((p2 - p1) / p1) * 100 if (((p2 - p1) / p1) * 100) < 0 else 0 for p2 in p2_l2l]
                
                change_prob_q2l = [((p2 - p1) / p1) * 100 for p2 in p2_q2l]
                change_prob_i2l = [((p2 - p1) / p1) * 100 for p2 in p2_i2l]
                change_prob_i2q = [((p2 - p1) / p1) * 100 for p2 in p2_i2q]
                change_prob_l2l = [((p2 - p1) / p1) * 100 for p2 in p2_l2l]

                # Add to the global df
                if current_word.lower() in dataframe_global:
                    dataframe_global[current_word.lower()]['image_to_last'] = [(a+b) for a,b in zip(change_prob_i2l_neg, dataframe_global[current_word.lower()]['image_to_last'])]
                    dataframe_global[current_word.lower()]['image_to_question'] = [(a+b) for a,b in zip(change_prob_i2q_neg, dataframe_global[current_word.lower()]['image_to_question'])]
                    dataframe_global[current_word.lower()]['question_to_last'] = [(a+b) for a,b in zip(change_prob_q2l_neg, dataframe_global[current_word.lower()]['question_to_last'])]
                    dataframe_global[current_word.lower()]['last_to_last'] = [(a+b) for a,b in zip(change_prob_l2l_neg, dataframe_global[current_word.lower()]['last_to_last'])]
                    dataframe_global[current_word.lower()]['count'] += 1
                    dataframe_global[current_word.lower()]['subword_count'] = subword_count
                else:
                    dataframe_global[current_word.lower()] = {}
                    dataframe_global[current_word.lower()]['image_to_last'] = change_prob_i2l_neg
                    dataframe_global[current_word.lower()]['image_to_question'] = change_prob_i2q_neg
                    dataframe_global[current_word.lower()]['question_to_last'] = change_prob_q2l_neg
                    dataframe_global[current_word.lower()]['last_to_last'] = change_prob_l2l_neg
                    dataframe_global[current_word.lower()]['count'] = 1
                    dataframe_global[current_word.lower()]['subword_count'] = subword_count
                    
        else:     
            continue

    for word in dataframe_global.keys():
        dataframe_global[word]['image_to_last'] = [c / dataframe_global[word]['count'] for c in dataframe_global[word]['image_to_last']]
        dataframe_global[word]['image_to_question'] = [c / dataframe_global[word]['count'] for c in dataframe_global[word]['image_to_question']]
        dataframe_global[word]['question_to_last'] = [c / dataframe_global[word]['count'] for c in dataframe_global[word]['question_to_last']]
        dataframe_global[word]['last_to_last'] = [c / dataframe_global[word]['count'] for c in dataframe_global[word]['last_to_last']]
    return dataframe_global
        
def create_df_llava(list_ids, answer_path):
    dataframe_global = {}
    if isinstance(list_ids, dict):
        ids = []
        for k in list_ids.keys():
            ids.extend(list_ids[k])
    else:
        ids = list_ids

    for id in ids: #for each sample
        path = answer_path + "/" + str(id) + ".json"
        if os.path.exists(path): 
            sample_probs = json.load(open(path))

            for idx, item in enumerate(sample_probs['full_attention']): #for each answer token
                #Beginning of a word
                current_word = item[0]
                p1= item[1]

                p2_i2q = np.array(sample_probs['image_to_question'][idx][1])
                p2_i2l = np.array(sample_probs['image_to_last'][idx][1])
                p2_q2l = np.array(sample_probs['question_to_last'][idx][1])
                p2_l2l = np.array(sample_probs['last_to_last'][idx][1])
                    
                # Compute information drop
                change_prob_q2l_neg = [((p2 - p1) / p1) * 100 if (((p2 - p1) / p1) * 100) < 0 else 0 for p2 in p2_q2l]
                change_prob_i2l_neg = [((p2 - p1) / p1) * 100 if (((p2 - p1) / p1) * 100) < 0 else 0 for p2 in p2_i2l]
                change_prob_i2q_neg = [((p2 - p1) / p1) * 100 if (((p2 - p1) / p1) * 100) < 0 else 0 for p2 in p2_i2q]
                change_prob_l2l_neg = [((p2 - p1) / p1) * 100 if (((p2 - p1) / p1) * 100) < 0 else 0 for p2 in p2_l2l]
                
                change_prob_q2l = [((p2 - p1) / p1) * 100 for p2 in p2_q2l]
                change_prob_i2l = [((p2 - p1) / p1) * 100 for p2 in p2_i2l]
                change_prob_i2q = [((p2 - p1) / p1) * 100 for p2 in p2_i2q]
                change_prob_l2l = [((p2 - p1) / p1) * 100 for p2 in p2_l2l]
                            
                # Add to the global df
                if current_word.lower() in dataframe_global:
                    dataframe_global[current_word.lower()]['image_to_last'] = [(a+b) for a,b in zip(change_prob_i2l_neg, dataframe_global[current_word.lower()]['image_to_last'])]
                    dataframe_global[current_word.lower()]['image_to_question'] = [(a+b) for a,b in zip(change_prob_i2q_neg, dataframe_global[current_word.lower()]['image_to_question'])]
                    dataframe_global[current_word.lower()]['question_to_last'] = [(a+b) for a,b in zip(change_prob_q2l_neg, dataframe_global[current_word.lower()]['question_to_last'])]
                    dataframe_global[current_word.lower()]['last_to_last'] = [(a+b) for a,b in zip(change_prob_l2l_neg, dataframe_global[current_word.lower()]['last_to_last'])]
                    dataframe_global[current_word.lower()]['count'] += 1
                else:
                    dataframe_global[current_word.lower()] = {}
                    dataframe_global[current_word.lower()]['image_to_last'] = change_prob_i2l_neg
                    dataframe_global[current_word.lower()]['image_to_question'] = change_prob_i2q_neg
                    dataframe_global[current_word.lower()]['question_to_last'] = change_prob_q2l_neg
                    dataframe_global[current_word.lower()]['last_to_last'] = change_prob_l2l_neg
                    dataframe_global[current_word.lower()]['count'] = 1
        else:     
            continue

    for word in dataframe_global.keys():
        dataframe_global[word]['image_to_last'] = [c / dataframe_global[word]['count'] for c in dataframe_global[word]['image_to_last']]
        dataframe_global[word]['image_to_question'] = [c / dataframe_global[word]['count'] for c in dataframe_global[word]['image_to_question']]
        dataframe_global[word]['question_to_last'] = [c / dataframe_global[word]['count'] for c in dataframe_global[word]['question_to_last']]
        dataframe_global[word]['last_to_last'] = [c / dataframe_global[word]['count'] for c in dataframe_global[word]['last_to_last']]
    return dataframe_global

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# COCO
# LLama Vision
df_global_llama = pd.DataFrame(create_df_llama(list_ids=list_ids_llama_coco, answer_path=answer_path_llama_coco)).T
df_global_llama = df_global_llama.reset_index().rename(columns={'index': 'word'})

df_global_llama['min_image_to_last'] = df_global_llama['image_to_last'].apply(min)
df_global_llama['min_image_to_question']= df_global_llama['image_to_question'].apply(min)

# LLaVA
df_global_llava = pd.DataFrame(create_df_llava(list_ids=list_ids_llava_coco, answer_path=answer_path_llava_coco)).T
df_global_llava = df_global_llava.reset_index().rename(columns={'index': 'word'})

df_global_llava['min_image_to_last'] = df_global_llava['image_to_last'].apply(min)
df_global_llava['min_image_to_question']= df_global_llava['image_to_question'].apply(min)

# Qwen
df_global_qwen = pd.DataFrame(create_df_llama(list_ids=list_ids_qwen_coco, answer_path=answer_path_qwen_coco)).T
df_global_qwen = df_global_qwen.reset_index().rename(columns={'index': 'word'})

df_global_qwen['min_image_to_last'] = df_global_qwen['image_to_last'].apply(min)
df_global_qwen['min_image_to_question']= df_global_qwen['image_to_question'].apply(min)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Filtering 
# Convert to numeric and coerce errors to NaN
df_global_llama['min_image_to_question'] = pd.to_numeric(df_global_llama['min_image_to_question'], errors='coerce')
df_global_llama['min_image_to_last'] = pd.to_numeric(df_global_llama['min_image_to_last'], errors='coerce')
df_global_llama['count'] = pd.to_numeric(df_global_llama['count'], errors='coerce')
df_llama_filtered = df_global_llama.dropna(subset=['min_image_to_question', 'min_image_to_last', 'count'])

df_global_llava['min_image_to_question'] = pd.to_numeric(df_global_llava['min_image_to_question'], errors='coerce')
df_global_llava['min_image_to_last'] = pd.to_numeric(df_global_llava['min_image_to_last'], errors='coerce')
df_global_llava['count'] = pd.to_numeric(df_global_llava['count'], errors='coerce')
df_llava_filtered = df_global_llava.dropna(subset=['min_image_to_question', 'min_image_to_last', 'count'])


df_global_qwen['min_image_to_question'] = pd.to_numeric(df_global_qwen['min_image_to_question'], errors='coerce')
df_global_qwen['min_image_to_last'] = pd.to_numeric(df_global_qwen['min_image_to_last'], errors='coerce')
df_global_qwen['count'] = pd.to_numeric(df_global_qwen['count'], errors='coerce')
df_qwen_filtered = df_global_qwen.dropna(subset=['min_image_to_question', 'min_image_to_last', 'count'])
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# soglia con primo percentile
df_global_filtered_llama = df_llama_filtered[
    (df_llama_filtered['count'] > df_llama_filtered['count'].quantile(0.9)) &  
    (df_llama_filtered['min_image_to_last'] < df_llama_filtered['min_image_to_last'].quantile(0.65))
].sort_values(by='min_image_to_question', ascending=True)

df_global_filtered_llama_no_flow = df_llama_filtered[
    (df_llama_filtered['count'] > df_llama_filtered['count'].quantile(0.9))
].sort_values(by='min_image_to_question', ascending=False)

df_global_filtered_llava = df_llava_filtered[
    (df_llava_filtered['count'] > df_llava_filtered['count'].quantile(0.9)) &  
    (df_llava_filtered['min_image_to_last'] < df_llava_filtered['min_image_to_last'].quantile(0.65))
].sort_values(by='min_image_to_question', ascending=True)

df_global_filtered_llava_no_flow = df_llava_filtered[
    (df_llava_filtered['count'] > df_llava_filtered['count'].quantile(0.9))
].sort_values(by='min_image_to_question', ascending=False)

df_global_filtered_qwen = df_qwen_filtered[
    (df_qwen_filtered['count'] > df_qwen_filtered['count'].quantile(0.9)) &  
    (df_qwen_filtered['min_image_to_last'] < df_qwen_filtered['min_image_to_last'].quantile(0.65))
].sort_values(by='min_image_to_question', ascending=True)

df_global_filtered_qwen_no_flow = df_qwen_filtered[
    (df_qwen_filtered['count'] > df_qwen_filtered['count'].quantile(0.9))
].sort_values(by='min_image_to_question', ascending=False)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot COCO words - LLaMA-Vision
# Etichette
row_labels = ['Content Words', 'Structural Words']
curve_labels = ['Last↛ Last', 'Image↛ Last', 'Text↛ Last', 'Image↛ Text']
colors = ['red', 'orange', 'green', 'purple']
keys = ['last_to_last', 'image_to_last', 'question_to_last', 'image_to_question']

subplot_labels = [
    ['showcasing', 'setting', 'features', 'situated', 'featuring'],
    ['image', 'of', 'depicts', 'to', 'a']
]

x_min = 0
x_max = 40
layers = np.arange(0, 40)
num_rows, num_cols = 2, 5
num_layers = len(layers)

# CA layer positions
ca_layers = [3, 8, 13, 18, 23, 28, 33, 38]
# Plot
fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 10), sharex=True, sharey=True)
for i in range(num_rows):
    for j in range(num_cols):
        ax = axes[i, j]
        # Row 0
        if i == 0 and j == 0:
            layers_change_probs = df_global_llama[df_global_llama['word']=='showcasing']

        elif i == 0 and j == 1:
            layers_change_probs = df_global_llama[df_global_llama['word']=='setting.']
        
        elif i == 0 and j == 2:
            layers_change_probs = df_global_llama[df_global_llama['word']=='features']
        
        elif i == 0 and j == 3:
            layers_change_probs = df_global_llama[df_global_llama['word']=='situated']

        elif i == 0 and j == 4:
            layers_change_probs = df_global_llama[df_global_llama['word']=='featuring']
        
        # Row 1
        elif i == 1 and j == 0:
            layers_change_probs = df_global_llama[df_global_llama['word']=='image']

        elif i == 1 and j == 1:
            layers_change_probs = df_global_llama[df_global_llama['word']=='of']
        
        elif i == 1 and j == 2:
            layers_change_probs = df_global_llama[df_global_llama['word']=='depicts']
        
        elif i == 1 and j == 3:
            layers_change_probs = df_global_llama[df_global_llama['word']=='to']

        elif i == 1 and j == 4:
            layers_change_probs = df_global_llama[df_global_llama['word']=='a']
        else:
            continue

        for k, color in enumerate(colors):
            ax.plot(layers, layers_change_probs[keys[k]].tolist()[0], color=color, label=curve_labels[k], linewidth=2, alpha=0.9)
        
        ax.xaxis.set_minor_locator(MultipleLocator(1))  
        ax.tick_params(axis='x', which='minor', length=5, direction='out', labelsize=18)
        ax.tick_params(axis='x', which='major', labelsize=18)
        ax.tick_params(axis='y', which='both', labelsize=18)
        ax.grid(True)
        ax.set_xlim(0, 40) 

        # Aggiungi la caption sotto ogni subplot
        ax.annotate(subplot_labels[i][j],
            xy=(0.5, 1.05),  # posizione sopra il grafico
            xycoords='axes fraction',
            ha='center',
            va='bottom',
            fontsize=28)

        #if i==1:  #LLaMA-Vision
            # Vertical lines for CA layers
        for ca in ca_layers:
            ax.axvline(x=ca, color='blue', linestyle=':')

        if j == 0:
            ax.set_ylabel('Probability Change (%)', fontsize=24)
        if i == 1:
            ax.set_xlabel('Layer', fontsize=24)

# Aggiungi etichette di riga
for ax, row in zip(axes[:, 0], row_labels):
    ax.annotate(row,
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - 10, 0),
                xycoords=ax.yaxis.label,
                textcoords='offset points',
                size=28,  # You can increase this number as needed
                ha='right',
                va='center',
                rotation=90)

# Legenda globale
handles = [plt.Line2D([0], [0], color=c, label=l) for c, l in zip(colors, curve_labels)]
handles.append(plt.Line2D([0], [0], color='blue', linestyle='--', label='CA Layers'))

fig.legend(handles=handles, 
           frameon=False,
           loc='upper center', 
           bbox_to_anchor=(0.5, 1.03),
           ncol=5, 
           fontsize=24)

plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.show()
plt.savefig(f'grafico_coco_words.pdf', format='pdf', dpi=600, bbox_inches='tight')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ChartQA - CoT complete
# LLama Vision
df_global_llama = pd.DataFrame(create_df_llama(list_ids=list_ids_llama_chartqa, answer_path=answer_path_llama_chartqa)).T
df_global_llama = df_global_llama.reset_index().rename(columns={'index': 'word'})

df_global_llama['min_image_to_last'] = df_global_llama['image_to_last'].apply(min)
df_global_llama['min_image_to_question']= df_global_llama['image_to_question'].apply(min)

df_global_filtered_llama = df_llama_filtered[
    (df_llama_filtered['count'] > df_llama_filtered['count'].quantile(0.9)) &  
    (df_llama_filtered['min_image_to_last'] < df_llama_filtered['min_image_to_last'].quantile(0.65))
].sort_values(by='min_image_to_question', ascending=True)
#examine, look, arrange, subtract, final

df_global_filtered_llama_no_flow = df_llama_filtered[
    (df_llama_filtered['count'] > df_llama_filtered['count'].quantile(0.9))
].sort_values(by='min_image_to_question', ascending=False)
# than, of, =, at, :, we, the

# LLaVA
df_global_llava = pd.DataFrame(create_df_llava(list_ids=list_ids_llava_chartqa, answer_path=answer_path_llava_chartqa)).T
df_global_llava = df_global_llava.reset_index().rename(columns={'index': 'word'})

df_global_llava['min_image_to_last'] = df_global_llava['image_to_last'].apply(min)
df_global_llava['min_image_to_question']= df_global_llava['image_to_question'].apply(min)

df_global_filtered_llava = df_llava_filtered[
    (df_llava_filtered['count'] > df_llava_filtered['count'].quantile(0.9)) &  
    (df_llava_filtered['min_image_to_last'] < df_llava_filtered['min_image_to_last'].quantile(0.65))
].sort_values(by='min_image_to_question', ascending=True)
#first, corresponding, shown, calculate, since, identify, look 

df_global_filtered_llava_no_flow = df_llava_filtered[
    (df_llava_filtered['count'] > df_llava_filtered['count'].quantile(0.9))
].sort_values(by='min_image_to_question', ascending=False)
# :, ,, we, to, of, in, for

# Qwen
df_global_qwen = pd.DataFrame(create_df_llama(list_ids=list_ids_qwen_chartqa, answer_path=answer_path_qwen_chartqa)).T
df_global_qwen = df_global_qwen.reset_index().rename(columns={'index': 'word'})

df_global_qwen['min_image_to_last'] = df_global_qwen['image_to_last'].apply(min)
df_global_qwen['min_image_to_question']= df_global_qwen['image_to_question'].apply(min)

df_global_filtered_qwen = df_qwen_filtered[
    (df_qwen_filtered['count'] > df_qwen_filtered['count'].quantile(0.9)) &  
    (df_qwen_filtered['min_image_to_last'] < df_qwen_filtered['min_image_to_last'].quantile(0.65))
].sort_values(by='min_image_to_question', ascending=True)

df_global_filtered_qwen_no_flow = df_qwen_filtered[
    (df_qwen_filtered['count'] > df_qwen_filtered['count'].quantile(0.9))
].sort_values(by='min_image_to_question', ascending=False)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ChartQA - Plot words for LLaMA-Vision
# Etichette
row_labels = ['Content Words', 'Structural Words']
curve_labels = ['Last↛ Last', 'Image↛ Last', 'Text↛ Last', 'Image↛ Text']
colors = ['red', 'orange', 'green', 'purple']
keys = ['last_to_last', 'image_to_last', 'question_to_last', 'image_to_question']

subplot_labels = [
    ['examine', 'look', 'arrange', 'subtract', 'final'],
    ['need', 'of', 'at', 'we', 'the']
]

x_min = 0
x_max = 40
layers = np.arange(0, 40)
num_rows, num_cols = 2, 5
num_layers = len(layers)

# CA layer positions
ca_layers = [3, 8, 13, 18, 23, 28, 33, 38]
# Plot
fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 10), sharex=True, sharey=True)
for i in range(num_rows):
    for j in range(num_cols):
        ax = axes[i, j]
        # Row 0
        if i == 0 and j == 0:
            layers_change_probs = df_global_llama[df_global_llama['word']=='examine']

        elif i == 0 and j == 1:
            layers_change_probs = df_global_llama[df_global_llama['word']=='look']
        
        elif i == 0 and j == 2:
            layers_change_probs = df_global_llama[df_global_llama['word']=='arrange']
        
        elif i == 0 and j == 3:
            layers_change_probs = df_global_llama[df_global_llama['word']=='subtract']

        elif i == 0 and j == 4:
            layers_change_probs = df_global_llama[df_global_llama['word']=='final']
        
        # Row 1
        elif i == 1 and j == 0:
            layers_change_probs = df_global_llama[df_global_llama['word']=='need']

        elif i == 1 and j == 1:
            layers_change_probs = df_global_llama[df_global_llama['word']=='of']
        
        elif i == 1 and j == 2:
            layers_change_probs = df_global_llama[df_global_llama['word']=='at']
        
        elif i == 1 and j == 3:
            layers_change_probs = df_global_llama[df_global_llama['word']=='we']

        elif i == 1 and j == 4:
            layers_change_probs = df_global_llama[df_global_llama['word']=='the']
        else:
            continue

        for k, color in enumerate(colors):
            ax.plot(layers, layers_change_probs[keys[k]].tolist()[0], color=color, label=curve_labels[k], linewidth=2, alpha=0.9)
        
        ax.xaxis.set_minor_locator(MultipleLocator(1))  
        ax.tick_params(axis='x', which='minor', length=5, direction='out', labelsize=18)
        ax.tick_params(axis='x', which='major', labelsize=18)
        ax.tick_params(axis='y', which='both', labelsize=18)
        ax.grid(True)
        ax.set_xlim(0, 40) 

        # Aggiungi la caption sotto ogni subplot
        ax.annotate(subplot_labels[i][j],
            xy=(0.5, 1.05),  # posizione sopra il grafico
            xycoords='axes fraction',
            ha='center',
            va='bottom',
            fontsize=28)

        #if i==1:  #LLaMA-Vision
            # Vertical lines for CA layers
        for ca in ca_layers:
            ax.axvline(x=ca, color='blue', linestyle=':')

        if j == 0:
            ax.set_ylabel('Probability Change (%)', fontsize=24)
        if i == 1:
            ax.set_xlabel('Layer', fontsize=24)

# Aggiungi etichette di riga
for ax, row in zip(axes[:, 0], row_labels):
    ax.annotate(row,
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - 10, 0),
                xycoords=ax.yaxis.label,
                textcoords='offset points',
                size=28,  # You can increase this number as needed
                ha='right',
                va='center',
                rotation=90) 

# Legenda globale
handles = [plt.Line2D([0], [0], color=c, label=l) for c, l in zip(colors, curve_labels)]
handles.append(plt.Line2D([0], [0], color='blue', linestyle='--', label='CA Layers'))

fig.legend(handles=handles, 
           frameon=False,
           loc='upper center', 
           bbox_to_anchor=(0.5, 1.03),
           ncol=5, 
           fontsize=24)


plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f'grafico_chart_words.pdf', format='pdf', dpi=600, bbox_inches='tight')

# %%
