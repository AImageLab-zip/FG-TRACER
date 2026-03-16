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
# CHARTQA COMPLETE
answer_path_chartqa_llama = "LLaMAVision/results/chartqa_complete"
datas_chartqa_llama = json.load(open('LLaMAVision/tracing_information_flow/dataset/chartqa/chartqa_dataset_correct_answers.json'))
list_ids_chartqa_llama = list(range(len(datas_chartqa_llama)))

answer_path_chartqa_llava = "LLaVA/results/chartqa"
datas_chartqa_llava = json.load(open('LLaVA/tracing_information_flow/datasets/chartqa/chartqa_dataset_correct_answers_complete.json'))
list_ids_chartqa_llava = list(range(len(datas_chartqa_llava)))

answer_path_chartqa_qwen = "Qwen2.5_VL/results/chartqa_complete"
datas_chartqa_qwen = json.load(open('Qwen2.5_VL/tracing_information_flow/datasets/chartqa/chartqa_dataset_correct_answers.json'))
list_ids_chartqa_qwen = list(range(len(datas_chartqa_qwen)))

# TextVQA
answer_path_textvqa_llama = "LLaMAVision/results/textvqa"
datas_textvqa_llama = json.load(open('LLaMAVision/tracing_information_flow/dataset/textvqa/textvqa_dataset_correct_answers.json'))
list_ids_textvqa_llama = json.load(open('LLaMAVision/tracing_information_flow/dataset/textvqa/ids_textvqa.json'))

answer_path_textvqa_llava = "LLaVA/results/textvqa"
datas_textvqa_llava = json.load(open('LLaVA/tracing_information_flow/datasets/textvqa/textvqa_dataset_correct_answers.json'))
list_ids_textvqa_llava = json.load(open('LLaVA/tracing_information_flow/datasets/textvqa/ids_textvqa.json'))

answer_path_textvqa_qwen = "Qwen2.5_VL/results/textvqa"
datas_textvqa_qwen = json.load(open('Qwen2.5_VL/tracing_information_flow/datasets/textvqa/textvqa_dataset_correct_answers.json'))
list_ids_textvqa_qwen = json.load(open('Qwen2.5_VL/tracing_information_flow/datasets/textvqa/ids_correct_textvqa.json'))

# COCO
answer_path_coco_llama = "LLaMAVision/results/coco"
datas_coco_llama = json.load(open('LLaMAVision/tracing_information_flow/dataset/coco/coco_dataset_correct_answers.json'))
list_ids_coco_llama = json.load(open('LLaMAVision/tracing_information_flow/dataset/coco/ids_coco2.json'))

answer_path_coco_llava = "LLaVA/results/coco"
datas_coco_llava = json.load(open('LLaVA/tracing_information_flow/datasets/coco/coco_dataset_correct_answers.json'))
list_ids_coco_llava = json.load(open('LLaVA/tracing_information_flow/datasets/coco/ids_coco.json'))

answer_path_coco_qwen = "Qwen2.5_VL/results/coco"
datas_coco_qwen = json.load(open('Qwen2.5_VL/tracing_information_flow/datasets/coco/coco_dataset_correct_answers.json'))
list_ids_coco_qwen = json.load(open('Qwen2.5_VL/tracing_information_flow/datasets/coco/ids_coco.json'))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eps=1e-15

def compute_curve(datas, list_ids, answer_path, FA=False, model=""):
    layers_change_probs = {}
    if 'textvqa' in answer_path:
        total_ids = []
        for _, ids in list_ids.items():
            total_ids.extend(ids)
        list_ids = set(total_ids)

    count_tot = 0
    for id in list_ids: #for each sample in a category
        path = answer_path + "/" + str(id) + ".json"
        if os.path.exists(path):
            sample_probs = json.load(open(path))

            if not FA: 
                len_sample =  len(sample_probs['full_attention'])
                p1 = 1
                for token, prob in sample_probs['full_attention']:
                    p1 = p1 * (prob ** (1 / len_sample))
                p1 = p1 + eps
            
            elif model == "LLaVA" and FA:  #LLaVA + FA
                start_idx = None
                for idx, (token, prob) in enumerate(sample_probs['full_attention']):
                    if token == 'ER' and sample_probs['last_to_last'][idx + 1][0] == ':':
                        start_idx = idx + 2
                if start_idx is not None:
                    len_sample =  len(sample_probs['full_attention'][start_idx:])
                    p1 = 1
                    for token, prob in sample_probs['full_attention'][start_idx:]:    
                        p1 = p1 * (prob ** (1 / len_sample))
                    p1 = p1 + eps
                else:
                    continue
            
            elif (model == "Qwen" or model == "LLaMA") and FA:  #Qwen/LLaMA Vision + FA
                # answer token length
                len_sample = len(datas[id]['final_tokens'])
                p1 = 1

                start_idx = len(datas[id]['thought_tokens'])
                for token, prob in sample_probs['full_attention'][start_idx:]:
                    p1 = p1 * (prob ** (1 / len_sample))
                p1 = p1 + eps
            
            else:
                raise NotImplementedError
            
            count_tot += 1

            ### Experiment 1
            if 'last_to_last' in sample_probs:
                if model == "LLaVA" or model == "LLaMA":
                    p2 = [1 for i in range(40)]
                elif model == "Qwen":
                    p2 = [1 for i in range(28)]
                else:
                    raise NotImplementedError

                if not FA:
                    for token, prob in sample_probs['last_to_last']:
                        p2 = [(a * (b**(1/len_sample))) for a, b in zip(p2, prob)]
                else:
                    for token, prob in sample_probs['last_to_last'][start_idx:]:
                        p2 = [(a * (b**(1/len_sample))) for a, b in zip(p2, prob)]

                if 'last_to_last' in layers_change_probs:
                    change_prob = [((p - p1) / p1) * 100 if p - p1 < 0 else 0 for p in p2] 
                    layers_change_probs['last_to_last'] = [(a + b) for a, b in zip(layers_change_probs['last_to_last'], change_prob)]
                else:
                    layers_change_probs['last_to_last'] = [((p - p1) / p1) * 100 if p - p1 < 0 else 0  for p in p2] 

            if 'question_to_last' in sample_probs:
                if model == "LLaVA" or model == "LLaMA":
                    p2 = [1 for i in range(40)]
                elif model == "Qwen":
                    p2 = [1 for i in range(28)]
                else:
                    raise NotImplementedError

                if not FA:
                    for token, prob in sample_probs['question_to_last']:
                        p2 = [(a * (b**(1/len_sample))) for a, b in zip(p2, prob)]
                else:
                    for token, prob in sample_probs['question_to_last'][start_idx:]:
                        p2 = [(a * (b**(1/len_sample))) for a, b in zip(p2, prob)]
                
                if 'question_to_last' in layers_change_probs:
                    change_prob = [((p - p1) / p1) * 100 if p - p1 < 0 else 0  for p in p2] 
                    layers_change_probs['question_to_last'] = [(a + b) for a, b in zip(layers_change_probs['question_to_last'], change_prob )]
                else:
                    layers_change_probs['question_to_last'] =  [((p - p1) / p1) * 100 if p - p1 < 0 else 0  for p in p2] 
        
            if 'image_to_last' in sample_probs:
                if model == "LLaVA" or model == "LLaMA":
                    p2 = [1 for i in range(40)]
                elif model == "Qwen":
                    p2 = [1 for i in range(28)]
                else:
                    raise NotImplementedError

                if not FA:
                    for token, prob in sample_probs['image_to_last']:
                        p2 = [(a * (b**(1/len_sample))) for a, b in zip(p2, prob)]
                else:
                    for token, prob in sample_probs['image_to_last'][start_idx:]:
                        p2 = [(a * (b**(1/len_sample))) for a, b in zip(p2, prob)]
            
                if 'image_to_last' in layers_change_probs:
                    change_prob = [((p - p1) / p1) * 100 if p - p1 < 0 else 0  for p in p2] 
                    layers_change_probs['image_to_last'] = [(a + b) for a, b in zip(layers_change_probs['image_to_last'], change_prob )]
                else:
                    layers_change_probs['image_to_last'] = [((p - p1) / p1) * 100 if p - p1 < 0 else 0 for p in p2] 
        
            ### Experiment 2
            if 'image_to_question' in sample_probs:
                if model == "LLaVA" or model == "LLaMA":
                    p2 = [1 for i in range(40)]
                elif model == "Qwen":
                    p2 = [1 for i in range(28)]
                else:
                    raise NotImplementedError

                if not FA:
                    for token, prob in sample_probs['image_to_question']:
                        p2 = [(a * (b**(1/len_sample))) for a, b in zip(p2, prob)]
                else:
                    for token, prob in sample_probs['image_to_question'][start_idx:]:
                        p2 = [(a * (b**(1/len_sample))) for a, b in zip(p2, prob)]

                if 'image_to_question' in layers_change_probs:
                    change_prob = [((p - p1) / p1) * 100 if p - p1 < 0 else 0  for p in p2] 
                    layers_change_probs['image_to_question'] = [(a + b) for a, b in zip(layers_change_probs['image_to_question'], change_prob )]
                else:
                    layers_change_probs['image_to_question'] = [((p - p1) / p1) * 100 if p - p1 < 0 else 0  for p in p2] 

        else:
            #print(f"file json {path} does not exist!!")
            continue

    for key in layers_change_probs.keys():
        layers_change_probs[key] = [x / count_tot for x in layers_change_probs[key]]
    print(count_tot)
    return layers_change_probs

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                               PLOT GENERAL INFORMATION FLOW
# Etichette
num_rows, num_cols = 3, 4

col_labels = ['TextVQA', 'COCO 2014', 'ChartQA: Complete CoT', 'ChartQA: Final Answer']
row_labels = ['LLaVA 1.5', 'LLaMA 3.2-Vision', 'Qwen 2.5-VL']
curve_labels = ['Last↛ Last', 'Image↛ Last', 'Text↛ Last', 'Image↛ Text']
colors = ['red', 'orange', 'green', 'purple']
keys = ['last_to_last', 'image_to_last', 'question_to_last', 'image_to_question']

# CA layer positions
ca_layers = [3, 8, 13, 18, 23, 28, 33, 38]

# Plot
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 12), sharey=True)

for i in range(num_rows):
    for j in range(num_cols):
        ax = axes[i, j]
        # Ror 0: LLAVA
        # TextVQA
        if i == 0 and j == 0:
            layers_change_probs = compute_curve(datas_textvqa_llava, list_ids_textvqa_llava, answer_path_textvqa_llava, model="LLaVA")
        # COCO
        elif i == 0 and j == 1:
            layers_change_probs = compute_curve(datas_coco_llava, list_ids_coco_llava, answer_path_coco_llava, model="LLaVA")
        # ChartQA
        elif i == 0 and j == 2:
            layers_change_probs =  compute_curve(datas_chartqa_llava, list_ids_chartqa_llava, answer_path_chartqa_llava, model="LLaVA")
        #CHARTQA FINAL ANSWER
        elif i == 0 and j == 3:
            layers_change_probs = compute_curve(datas_chartqa_llava, list_ids_chartqa_llava, answer_path_chartqa_llava, FA=True, model="LLaVA")
        
        # Row 1: LLaMA Vision
        # TextVQA
        elif i == 1 and j == 0:
            layers_change_probs = compute_curve(datas_textvqa_llama, list_ids_textvqa_llama, answer_path_textvqa_llama, model="LLaMA")
        # COCO
        elif i == 1 and j == 1:
            layers_change_probs = compute_curve(datas_coco_llama, list_ids_coco_llama, answer_path_coco_llama, model="LLaMA")
        # ChartQA
        elif i == 1 and j == 2:
            layers_change_probs = compute_curve(datas_chartqa_llama, list_ids_chartqa_llama, answer_path_chartqa_llama, model="LLaMA")
        # CHARTQA FINAL ANSWER
        elif i == 1 and j == 3:
            layers_change_probs = compute_curve(datas_chartqa_llama, list_ids_chartqa_llama, answer_path_chartqa_llama, FA=True, model="LLaMA")
        
        # Row 2: Qwen-VL
        # TextVQA
        elif i == 2 and j == 0:
            layers_change_probs = compute_curve(datas_textvqa_qwen, list_ids_textvqa_qwen, answer_path_textvqa_qwen, model="Qwen")
        # COCO
        elif i == 2 and j == 1:
            layers_change_probs = compute_curve(datas_coco_qwen, list_ids_coco_qwen, answer_path_coco_qwen, model="Qwen")
        # ChartQA
        elif i == 2 and j == 2:
            layers_change_probs = compute_curve(datas_chartqa_qwen, list_ids_chartqa_qwen, answer_path_chartqa_qwen, model="Qwen")
        # CHARTQA FINAL ANSWER
        elif i == 2 and j == 3:
            layers_change_probs = compute_curve(datas_chartqa_qwen, list_ids_chartqa_qwen, answer_path_chartqa_qwen, FA=True, model="Qwen")
        
        else:
            continue
        
        if i == 2:  #Qwen-VL has only 28 layers
            layers = np.arange(0, 28)
        elif i == 0 or i == 1:  #LLaVA and LLaMA-Vision have 40 layers
            layers = np.arange(0, 40)

        # Plot each curve
        for k, color in enumerate(colors):
            ax.plot(layers, layers_change_probs[keys[k]], color=color, label=curve_labels[k], linewidth=2, alpha=0.9)
        
        ax.xaxis.set_minor_locator(MultipleLocator(1))  
        ax.tick_params(axis='x', which='minor', length=5, direction='out', labelsize=15)
        ax.tick_params(axis='x', which='major', labelsize=12)
        ax.tick_params(axis='y', which='both', labelsize=12)

        ax.grid(True)

        if i==0 or i==1:  #LLaVA
            ax.set_xlim(0, 40) 
        elif i==2:  #Qwen-VL
            ax.set_xlim(0, 28)

        if i==1:  #LLaMA-Vision
            # Vertical lines for CA layers
            for ca in ca_layers:
                ax.axvline(x=ca, color='blue', linestyle=':')

        if j == 0:
            ax.set_ylabel('Change in probability (%)', fontsize=18)
        if i == 2:
            ax.set_xlabel('Layer', fontsize=18)

# Aggiungi etichette di riga
for ax, row in zip(axes[:, 0], row_labels):
    ax.annotate(row,
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - 10, 0),
                xycoords=ax.yaxis.label,
                textcoords='offset points',
                size=20,  # You can increase this number as needed
                ha='right',
                va='center',
                rotation=90)
    
for ax, col in zip(axes[0], col_labels):  
    ax.set_title(col, fontsize=20)  

# Legenda globale
handles = [plt.Line2D([0], [0], color=c, label=l) for c, l in zip(colors, curve_labels)]
handles.append(plt.Line2D([0], [0], color='blue', linestyle='--', label='CA Layers'))

fig.legend(handles=handles, 
           frameon=False,
           loc='upper center', 
           bbox_to_anchor=(0.5, 1),
           ncol=5, 
           fontsize=18, 
           alignment='center')


plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.show()
plt.savefig(f'grafico_generale.pdf', format='pdf', dpi=600, bbox_inches='tight')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_curve_not_normalized(datas, list_ids, answer_path, FA=False, model=""):
    layers_change_probs = {}
    if 'textvqa' in answer_path:
        total_ids = []
        count = 0
        for category, ids in list_ids.items():
            total_ids.extend(ids)
        list_ids = set(total_ids)

    count_tot = 0
    for id in list_ids: #for each sample in a category
        path = answer_path + "/" + str(id) + ".json"
        if os.path.exists(path):
            #print(path)
            sample_probs = json.load(open(path))
            # answer token length
            if not FA: 
                len_sample =  len(sample_probs['full_attention'])
                p1 = 1
                for token, prob in sample_probs['full_attention']:
                    p1 = p1 * prob
                p1 = p1 + eps
            
            elif model == "LLaVA" and FA:
                start_idx = None
                for idx, (token, prob) in enumerate(sample_probs['full_attention']):
                    if token == 'ER' and sample_probs['last_to_last'][idx + 1][0] == ':':
                            start_idx = idx + 2
                if start_idx is not None:
                    p1 = 1
                    for token, prob in sample_probs['full_attention'][start_idx:]:    
                        p1 = p1 * prob
                    p1 = p1 + eps
                else:
                    continue

            elif (model == "Qwen" or model == "LLaMA"):  #LLaMA Vision/Qwen2.5-VL + FA                
                # answer token length
                p1 = 1
                start_idx = len(datas[id]['thought_tokens'])
                for token, prob in sample_probs['full_attention'][start_idx:]:
                    p1 = p1 * prob 
                p1 = p1 + eps
            else:
                raise NotImplementedError
            
            #if p1 < 0.5:
            #    continue
            
            count_tot += 1

            ### Experiment 1
            if 'last_to_last' in sample_probs:
                if model == "LLaVA" or model == "LLaMA":
                    p2 = [1 for i in range(40)]
                elif model == "Qwen":
                    p2 = [1 for i in range(28)]
                else:
                    raise NotImplementedError
                
                if not FA:
                    for token, prob in sample_probs['last_to_last']:
                        p2 = [(a * b) for a, b in zip(p2, prob)]
                else:
                    for token, prob in sample_probs['last_to_last'][start_idx:]:
                        p2 = [(a * b) for a, b in zip(p2, prob)]

                if 'last_to_last' in layers_change_probs:
                    change_prob = [((p - p1) / p1) * 100 if p - p1 < 0 else 0 for p in p2] #[((p**(1/len_sample) - p1**(1/len_sample)) / p1**(1/len_sample)) * 100 for p in p2]
                    layers_change_probs['last_to_last'] = [(a + b) for a, b in zip(layers_change_probs['last_to_last'], change_prob)]
                else:
                    layers_change_probs['last_to_last'] = [((p - p1) / p1) * 100 if p - p1 < 0 else 0  for p in p2] #[((p**(1/len_sample) - p1**(1/len_sample)) / p1**(1/len_sample)) * 100 for p in p2]

            if 'question_to_last' in sample_probs:
                if model == "LLaVA" or model == "LLaMA":
                    p2 = [1 for i in range(40)]
                elif model == "Qwen":
                    p2 = [1 for i in range(28)]
                else:
                    raise NotImplementedError
                
                if not FA:
                    for token, prob in sample_probs['question_to_last']:
                        p2 = [(a * b) for a, b in zip(p2, prob)]
                else:
                    for token, prob in sample_probs['question_to_last'][start_idx:]:
                        p2 = [(a * b) for a, b in zip(p2, prob)]

                if 'question_to_last' in layers_change_probs:
                    change_prob = [((p - p1) / p1) * 100 if p - p1 < 0 else 0  for p in p2] #[((p**(1/len_sample) - p1**(1/len_sample)) / p1**(1/len_sample)) * 100 for p in p2]
                    layers_change_probs['question_to_last'] = [(a + b) for a, b in zip(layers_change_probs['question_to_last'], change_prob )]
                else:
                    layers_change_probs['question_to_last'] =  [((p - p1) / p1) * 100 if p - p1 < 0 else 0  for p in p2] #[((p**(1/len_sample) - p1**(1/len_sample)) / p1**(1/len_sample)) * 100 for p in p2]
        
            if 'image_to_last' in sample_probs:
                if model == "LLaVA" or model == "LLaMA":
                    p2 = [1 for i in range(40)]
                elif model == "Qwen":
                    p2 = [1 for i in range(28)]
                else:
                    raise NotImplementedError
                
                if not FA:
                    for token, prob in sample_probs['image_to_last']:
                        p2 = [(a * b) for a, b in zip(p2, prob)]
                else:
                    for token, prob in sample_probs['image_to_last'][start_idx:]:
                        p2 = [(a * b) for a, b in zip(p2, prob)]
            
                if 'image_to_last' in layers_change_probs:
                    change_prob = [((p - p1) / p1) * 100 if p - p1 < 0 else 0  for p in p2] #[((p**(1/len_sample) - p1**(1/len_sample)) / p1**(1/len_sample)) * 100 for p in p2]
                    layers_change_probs['image_to_last'] = [(a + b) for a, b in zip(layers_change_probs['image_to_last'], change_prob )]
                else:
                    layers_change_probs['image_to_last'] = [((p - p1) / p1) * 100 if p - p1 < 0 else 0 for p in p2] #[((p**(1/len_sample) - p1**(1/len_sample)) / p1**(1/len_sample)) * 100 for p in p2]
        
            ### Experiment 2
            if 'image_to_question' in sample_probs:
                if model == "LLaVA" or model == "LLaMA":
                    p2 = [1 for i in range(40)]
                elif model == "Qwen":
                    p2 = [1 for i in range(28)]
                else:
                    raise NotImplementedError
                
                if not FA:
                    for token, prob in sample_probs['image_to_question']:
                        p2 = [(a * b) for a, b in zip(p2, prob)]
                else:
                    for token, prob in sample_probs['image_to_question'][start_idx:]:
                        p2 = [(a * b) for a, b in zip(p2, prob)]

                if 'image_to_question' in layers_change_probs:
                    change_prob = [((p - p1) / p1) * 100 if p - p1 < 0 else 0  for p in p2] #[((p**(1/len_sample) - p1**(1/len_sample)) / p1**(1/len_sample)) * 100 for p in p2]
                    layers_change_probs['image_to_question'] = [(a + b) for a, b in zip(layers_change_probs['image_to_question'], change_prob )]
                else:
                    layers_change_probs['image_to_question'] = [((p - p1) / p1) * 100 if p - p1 < 0 else 0  for p in p2] #[((p**(1/len_sample) - p1**(1/len_sample)) / p1**(1/len_sample)) * 100 for p in p2]

        else:
            #print(f"file json {path} does not exist!!")
            continue

    for key in layers_change_probs.keys():
        layers_change_probs[key] = [x / count_tot for x in layers_change_probs[key]]

    return layers_change_probs
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                             PLOT GENERAL INFORMATION FLOW NOT BALANCED (on LLaMA Vision)
# Etichette
x_min = 0
x_max = 40
layers = np.arange(0, 40)
num_rows, num_cols = 2, 4
num_layers = len(layers)
col_labels = ['TextVQA', 'COCO 2014', 'ChartQA: Complete CoT', 'ChartQA: Final Answer']
row_labels = ['w/o normalization', 'w/ normalization']
curve_labels = ['Last↛ Last', 'Image↛ Last', 'Text↛ Last', 'Image↛ Text']
colors = ['red', 'orange', 'green', 'purple']
keys = ['last_to_last', 'image_to_last', 'question_to_last', 'image_to_question']

# CA layer positions
ca_layers = [3, 8, 13, 18, 23, 28, 33, 38]

# Plot
fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 8), sharex=True, sharey=True)

for i in range(num_rows):
    for j in range(num_cols):
        ax = axes[i, j]
        # Ror 0: LLaMA Vision not balanced
        # TextVQA
        if i == 0 and j == 0:
            layers_change_probs = compute_curve_not_normalized(datas_textvqa_llama, list_ids_textvqa_llama, answer_path_textvqa_llama, model="LLaMA")
        # COCO
        elif i == 0 and j == 1:
            layers_change_probs = compute_curve_not_normalized(datas_coco_llama, list_ids_coco_llama, answer_path_coco_llama, model="LLaMA")
        # ChartQA
        elif i == 0 and j == 2:
            layers_change_probs =  compute_curve_not_normalized(datas_chartqa_llama, list_ids_chartqa_llama, answer_path_chartqa_llama, model="LLaMA")
        #CHARTQA FINAL ANSWER
        elif i == 0 and j == 3:
            layers_change_probs = compute_curve_not_normalized(datas_chartqa_llama, list_ids_chartqa_llama, answer_path_chartqa_llama, FA=True, model="LLaMA")
        
        
        # Row 1: LLaMA Vision balanced
        # TextVQA
        elif i == 1 and j == 0:
            layers_change_probs = compute_curve(datas_textvqa_llama, list_ids_textvqa_llama, answer_path_textvqa_llama, model="LLaMA")
        # COCO
        elif i == 1 and j == 1:
            layers_change_probs = compute_curve(datas_coco_llama, list_ids_coco_llama, answer_path_coco_llama, model="LLaMA")
        # ChartQA
        elif i == 1 and j == 2:
            layers_change_probs = compute_curve(datas_chartqa_llama, list_ids_chartqa_llama, answer_path_chartqa_llama, model="LLaMA")
        # CHARTQA FINAL ANSWER
        elif i == 1 and j == 3:
            layers_change_probs = compute_curve(datas_chartqa_llama, list_ids_chartqa_llama, answer_path_chartqa_llama, FA=True, model="LLaMA")
        
        else:
            continue

        for k, color in enumerate(colors):
            ax.plot(layers, layers_change_probs[keys[k]], color=color, label=curve_labels[k], linewidth=2, alpha=0.9)
        
        ax.xaxis.set_minor_locator(MultipleLocator(1))  
        ax.tick_params(axis='x', which='minor', length=5, direction='out', labelsize=15)
        ax.tick_params(axis='x', which='major', labelsize=12)
        ax.tick_params(axis='y', which='both', labelsize=12)

        ax.grid(True)
        ax.set_xlim(0, 40) 

        #if i==1:  #LLaMA-Vision
            # Vertical lines for CA layers
        for ca in ca_layers:
            ax.axvline(x=ca, color='blue', linestyle=':')

        #ax.set_title(f'({chr(97 + i + j)}) {el_labels[i + j]}', fontsize=18)

        if j == 0:
            ax.set_ylabel('Change in probability (%)', fontsize=18)
        if i == 1:
            ax.set_xlabel('Layer', fontsize=18)

# Aggiungi etichette di riga
for ax, row in zip(axes[:, 0], row_labels):
    ax.annotate(row,
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - 10, 0),
                xycoords=ax.yaxis.label,
                textcoords='offset points',
                size=20,  # You can increase this number as needed
                ha='right',
                va='center',
                rotation=90)
    
for ax, col in zip(axes[0], col_labels):  
    ax.set_title(col, fontsize=20)  

# Legenda globale
handles = [plt.Line2D([0], [0], color=c, label=l) for c, l in zip(colors, curve_labels)]
handles.append(plt.Line2D([0], [0], color='blue', linestyle='--', label='CA Layers'))

fig.legend(handles=handles,
           frameon=False, 
           loc='upper center', 
           bbox_to_anchor=(0.5, 1.02),
           ncol=5, 
           fontsize=18, 
           alignment='center')

plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.show()
plt.savefig(f'grafico_norm_factor.pdf', format='pdf', dpi=600, bbox_inches='tight')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                       TextVQA Window Sizes on LLaMA Vision
datas_textvqa_llama = json.load(open('LLaMAVision/tracing_information_flow/dataset/textvqa/textvqa_dataset_correct_answers.json'))
list_ids_textvqa_llama = json.load(open('LLaMAVision/tracing_information_flow/dataset/textvqa/ids_textvqa.json'))
answer_path_textvqa_k_3 = "LLaMAVision/results/textvqa_k_3"
answer_path_textvqa_k_5 = "LLaMAVision/results/textvqa_k_5"
answer_path_textvqa_k_7 = "LLaMAVision/results/textvqa_k_7"
answer_path_textvqa_k_9 = "LLaMAVision/results/textvqa"
answer_path_textvqa_k_11 = "LLaMAVision/results/textvqa_k_11"
answer_path_textvqa_k_15 = "LLaMAVision/results/textvqa_k_15"
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x_min = 0
x_max = 40
layers = np.arange(0, 40)
num_rows, num_cols = 2, 3
num_layers = len(layers)

subplot_labels = [
    ['Window 3', 'Window 5', 'Window 7'],
    ['Window 9', 'Window 11', 'Window 15']
]
curve_labels = ['Last↛ Last', 'Image↛ Last', 'Text↛ Last', 'Image↛ Text']
colors = ['red', 'orange', 'green', 'purple']
keys = ['last_to_last', 'image_to_last', 'question_to_last', 'image_to_question']

# CA layer positions
ca_layers = [3, 8, 13, 18, 23, 28, 33, 38]

# Plot
fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 8), sharex=True, sharey=True)

for i in range(num_rows):
    for j in range(num_cols):
        ax = axes[i, j]
        # Row 0
        # K = 3
        if i == 0 and j == 0:
            layers_change_probs = compute_curve(datas_textvqa_llama, list_ids_textvqa_llama, answer_path_textvqa_k_3, model="LLaMA")
        # K = 5
        elif i == 0 and j == 1:
            layers_change_probs = compute_curve(datas_textvqa_llama, list_ids_textvqa_llama, answer_path_textvqa_k_5, model="LLaMA")
        # K = 7
        elif i == 0 and j == 2:
            layers_change_probs = compute_curve(datas_textvqa_llama, list_ids_textvqa_llama, answer_path_textvqa_k_7, model="LLaMA")
        
        # Row 1
        # K = 9
        elif i == 1 and j == 0:
            layers_change_probs = compute_curve(datas_textvqa_llama, list_ids_textvqa_llama, answer_path_textvqa_k_9, model="LLaMA")
        # K = 11
        elif i == 1 and j == 1:
            layers_change_probs = compute_curve(datas_textvqa_llama, list_ids_textvqa_llama, answer_path_textvqa_k_11, model="LLaMA")
        # K = 15
        elif i == 1 and j == 2:
            layers_change_probs = compute_curve(datas_textvqa_llama, list_ids_textvqa_llama, answer_path_textvqa_k_15, model="LLaMA")

        else:
            continue

        for k, color in enumerate(colors):
            ax.plot(layers, layers_change_probs[keys[k]], color=color, label=curve_labels[k], linewidth=2, alpha=0.9)
        
        ax.xaxis.set_minor_locator(MultipleLocator(1))  
        ax.tick_params(axis='x', which='minor', length=5, direction='out', labelsize=15)
        ax.tick_params(axis='x', which='major', labelsize=12)
        ax.tick_params(axis='y', which='both', labelsize=12)

        ax.grid(True)
        ax.set_xlim(0, 40) 

        # Aggiungi la caption sotto ogni subplot
        ax.annotate(subplot_labels[i][j],
            xy=(0.5, 1.05),  # posizione sopra il grafico
            xycoords='axes fraction',
            ha='center',
            va='bottom',
            fontsize=20)
        
        for ca in ca_layers:
            ax.axvline(x=ca, color='blue', linestyle=':')

        #ax.set_title(f'({chr(97 + i + j)}) {el_labels[i + j]}', fontsize=18)

        if j == 0:
            ax.set_ylabel('Probability Change (%)', fontsize=18)
        if i == 1:
            ax.set_xlabel('Layer', fontsize=18)

# Legenda globale
handles = [plt.Line2D([0], [0], color=c, label=l) for c, l in zip(colors, curve_labels)]
handles.append(plt.Line2D([0], [0], color='blue', linestyle='--', label='CA Layers'))

fig.legend(handles=handles,
           frameon=False, 
           loc='upper center', 
           bbox_to_anchor=(0.5, 1.02),
           ncol=5, 
           fontsize=18, 
           alignment='center')


plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.show()
plt.savefig(f'grafico_k_ablation.pdf', format='pdf', dpi=600, bbox_inches='tight')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                                   COCO Window Sizes on LLaMA Vision
datas_coco_llama = json.load(open('LLaMAVision/tracing_information_flow/dataset/coco/coco_dataset_correct_answers.json'))
list_ids_coco_llama = json.load(open('LLaMAVision/tracing_information_flow/dataset/coco/ids_coco2.json'))
answer_path_coco_k_3 = "LLaMAVision/results/coco_k3"
answer_path_coco_k_5 = "LLaMAVision/results/coco_k5"
answer_path_coco_k_7 = "LLaMAVision/results/coco_k7"
answer_path_coco_k_9 = "LLaMAVision/results/coco"
answer_path_coco_k_11 = "LLaMAVision/results/coco_k11"
answer_path_coco_k_15 = "LLaMAVision/results/coco_k15"
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x_min = 0
x_max = 40
layers = np.arange(0, 40)
num_rows, num_cols = 2, 3
num_layers = len(layers)

subplot_labels = [
    ['Window 3', 'Window 5', 'Window 7'],
    ['Window 9', 'Window 11', 'Window 15']
]
curve_labels = ['Last↛ Last', 'Image↛ Last', 'Text↛ Last', 'Image↛ Text']
colors = ['red', 'orange', 'green', 'purple']
keys = ['last_to_last', 'image_to_last', 'question_to_last', 'image_to_question']

# CA layer positions
ca_layers = [3, 8, 13, 18, 23, 28, 33, 38]

# Plot
fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 8), sharex=True, sharey=True)

for i in range(num_rows):
    for j in range(num_cols):
        ax = axes[i, j]
        # Ror 0
        # K = 3
        if i == 0 and j == 0:
            layers_change_probs = compute_curve(datas_coco_llama, list_ids_coco_llama, answer_path_coco_k_3, model="LLaMA")
        # K = 5
        elif i == 0 and j == 1:
            layers_change_probs = compute_curve(datas_coco_llama, list_ids_coco_llama, answer_path_coco_k_5, model="LLaMA")
        # K = 7
        elif i == 0 and j == 2:
            layers_change_probs = compute_curve(datas_coco_llama, list_ids_coco_llama, answer_path_coco_k_7, model="LLaMA")
        
        # Row 1
        # K = 9
        elif i == 1 and j == 0:
            layers_change_probs = compute_curve(datas_coco_llama, list_ids_coco_llama, answer_path_coco_k_9, model="LLaMA")
        # K = 11
        elif i == 1 and j == 1:
            layers_change_probs = compute_curve(datas_coco_llama, list_ids_coco_llama, answer_path_coco_k_11, model="LLaMA")
        # K = 15
        elif i == 1 and j == 2:
            layers_change_probs = compute_curve(datas_coco_llama, list_ids_coco_llama, answer_path_coco_k_15, model="LLaMA")

        else:
            continue

        for k, color in enumerate(colors):
            ax.plot(layers, layers_change_probs[keys[k]], color=color, label=curve_labels[k], linewidth=2, alpha=0.9)
        
        ax.xaxis.set_minor_locator(MultipleLocator(1))  
        ax.tick_params(axis='x', which='minor', length=5, direction='out', labelsize=15)
        ax.tick_params(axis='x', which='major', labelsize=12)
        ax.tick_params(axis='y', which='both', labelsize=12)

        ax.grid(True)
        ax.set_xlim(0, 40) 

        # Aggiungi la caption sotto ogni subplot
        ax.annotate(subplot_labels[i][j],
            xy=(0.5, 1.05),  # posizione sopra il grafico
            xycoords='axes fraction',
            ha='center',
            va='bottom',
            fontsize=20)
        
        for ca in ca_layers:
            ax.axvline(x=ca, color='blue', linestyle=':')

        #ax.set_title(f'({chr(97 + i + j)}) {el_labels[i + j]}', fontsize=18)

        if j == 0:
            ax.set_ylabel('Probability Change (%)', fontsize=18)
        if i == 1:
            ax.set_xlabel('Layer', fontsize=18)

# Legenda globale
handles = [plt.Line2D([0], [0], color=c, label=l) for c, l in zip(colors, curve_labels)]
handles.append(plt.Line2D([0], [0], color='blue', linestyle='--', label='CA Layers'))

fig.legend(handles=handles,
           frameon=False, 
           loc='upper center', 
           bbox_to_anchor=(0.5, 1.02),
           ncol=5, 
           fontsize=18, 
           alignment='center')


plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.show()
plt.savefig(f'LLaMAVision/tracing_information_flow/visualization/grafico_k_ablation_coco.pdf', format='pdf', dpi=600, bbox_inches='tight')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                                                       PLOT WRONG TEXTVQA
#LLaMA Vision
datas_textvqa_llama = json.load(open('LLaMAVision/tracing_information_flow/dataset/textvqa/textvqa_dataset_correct_answers.json'))
list_ids_textvqa_llama = json.load(open('LLaMAVision/tracing_information_flow/dataset/textvqa/ids_textvqa.json'))
answer_path_textvqa_llama = "LLaMAVision/results/textvqa"

datas_textvqa_llama_wrong = json.load(open('LLaMAVision/tracing_information_flow/dataset/textvqa/textvqa_dataset_wrong_answers.json'))
list_ids_textvqa_llama_wrong = json.load(open('LLaMAVision/tracing_information_flow/dataset/textvqa/ids_wrong_textvqa.json'))
answer_path_textvqa_llama_wrong = "LLaMAVision/results/textvqa_wrong"

# LLaVA
datas_textvqa_llava_wrong = json.load(open('LLaVA/tracing_information_flow/datasets/textvqa/textvqa_dataset_wrong_answers.json'))
list_ids_textvqa_llava_wrong = json.load(open('LLaVA/tracing_information_flow/datasets/textvqa/ids_textvqa_wrong.json'))
answer_path_textvqa_llava_wrong = "LLaVA/results/textvqa_wrong"

datas_textvqa_llava = json.load(open('LLaVA/tracing_information_flow/datasets/textvqa/textvqa_dataset_correct_answers.json'))
list_ids_textvqa_llava = json.load(open('LLaVA/tracing_information_flow/datasets/textvqa/ids_textvqa.json'))
answer_path_textvqa_llava = "LLaVA/results/textvqa"


# Qwen 2.5-VL
datas_textvqa_qwen_wrong = json.load(open('Qwen2.5_VL/tracing_information_flow/datasets/textvqa/textvqa_dataset_wrong_answers.json'))
list_ids_textvqa_qwen_wrong = json.load(open('Qwen2.5_VL/tracing_information_flow/datasets/textvqa/ids_wrong_textvqa.json'))
answer_path_textvqa_qwen_wrong = "Qwen2.5_VL/results/textvqa_wrong"

datas_textvqa_qwen = json.load(open('Qwen2.5_VL/tracing_information_flow/datasets/textvqa/textvqa_dataset_correct_answers.json'))
list_ids_textvqa_qwen = json.load(open('Qwen2.5_VL/tracing_information_flow/datasets/textvqa/ids_correct_textvqa.json'))
answer_path_textvqa_qwen = "Qwen2.5_VL/results/textvqa"
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x_min = 0
x_max = 40
num_rows, num_cols = 3, 2

layers = np.arange(0, 40)

col_labels = ['Correct Predictions', 'Incorrect Predictions']
row_labels = ['LLaVA 1.5', 'LLaMA 3.2-Vision', 'Qwen 2.5-VL']
curve_labels = ['Last↛ Last', 'Image↛ Last', 'Text↛ Last', 'Image↛ Text']
colors = ['red', 'orange', 'green', 'purple']
keys = ['last_to_last', 'image_to_last', 'question_to_last', 'image_to_question']

# CA layer positions
ca_layers = [3, 8, 13, 18, 23, 28, 33, 38]

# Plot
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 12), sharey=True)

for i in range(num_rows):
    for j in range(num_cols):
        ax = axes[i, j]
        # Ror 0: LLAVA
        # TextVQA correct
        if i == 0 and j == 0:
            layers_change_probs = compute_curve(datas_textvqa_llava, list_ids_textvqa_llava, answer_path_textvqa_llava, model="LLaVA")
        
        # TextVQA wrong
        elif i == 0 and j == 1:
            layers_change_probs = compute_curve(datas_textvqa_llava_wrong, list_ids_textvqa_llava_wrong, answer_path_textvqa_llava_wrong, model="LLaVA")
       
        
        # Row 1: LLaMA Vision
        # TextVQA correct
        elif i == 1 and j == 0:
            layers_change_probs = compute_curve(datas_textvqa_llama, list_ids_textvqa_llama, answer_path_textvqa_llama, model="LLaMA")
        # TextVQA wrong
        elif i == 1 and j == 1:
            layers_change_probs = compute_curve(datas_textvqa_llama_wrong, list_ids_textvqa_llama_wrong, answer_path_textvqa_llama_wrong, model="LLaMA")
        
        # Row 2: Qwen Vision
        # TextVQA correct
        elif i == 2 and j == 0:
            layers_change_probs = compute_curve(datas_textvqa_qwen, list_ids_textvqa_qwen, answer_path_textvqa_qwen, model="Qwen")
        # TextVQA wrong
        elif i == 2 and j == 1:
            layers_change_probs = compute_curve(datas_textvqa_qwen_wrong, list_ids_textvqa_qwen_wrong, answer_path_textvqa_qwen_wrong, model="Qwen")
        else:
            continue

        # plot curves
        # --- dynamic x based on returned curve length ---
        n_layers = len(layers_change_probs[keys[0]])
        x = np.arange(n_layers)

        for k, color, label in zip(keys, colors, curve_labels):
            y40 = layers_change_probs[k] #pad_to_40(layers_change_probs[k], mode="nan")
            ax.plot(x, y40, color=color, label=label, linewidth=2, alpha=0.9)

        ax.xaxis.set_minor_locator(MultipleLocator(1))  
        ax.tick_params(axis='x', which='minor', length=5, direction='out', labelsize=15)
        ax.tick_params(axis='x', which='major', labelsize=12)
        ax.tick_params(axis='y', which='both', labelsize=12)

        ax.grid(True)
        if i == 2:  # Qwen row
            ax.set_xlim(0, 27)   # optional, already happening
        else:
            ax.set_xlim(0, 39)
 

        if i==1:  #LLaMA-Vision
            # Vertical lines for CA layers
            for ca in ca_layers:
                ax.axvline(x=ca, color='blue', linestyle=':')

        if j == 0:
            ax.set_ylabel('Change in probability (%)', fontsize=18)
        if i == 2:
            ax.set_xlabel('Layer', fontsize=18)

# Aggiungi etichette di riga
for ax, row in zip(axes[:, 0], row_labels):
    ax.annotate(row,
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - 10, 0),
                xycoords=ax.yaxis.label,
                textcoords='offset points',
                size=20,  # You can increase this number as needed
                ha='right',
                va='center',
                rotation=90)
    
for ax, col in zip(axes[0], col_labels):  
    ax.set_title(col, fontsize=20)  

# Legenda globale
handles = [plt.Line2D([0], [0], color=c, label=l) for c, l in zip(colors, curve_labels)]
handles.append(plt.Line2D([0], [0], color='blue', linestyle='--', label='CA Layers'))

fig.legend(handles=handles, 
           frameon=False,
           loc='upper center', 
           bbox_to_anchor=(0.5, 1),
           ncol=5, 
           fontsize=18, 
           alignment='center')


plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.show()
plt.savefig(f'grafico_text_cw.pdf', format='pdf', dpi=600, bbox_inches='tight')