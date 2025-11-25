import sys
sys.path.append('./')
import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import torch
from PIL import Image
from generation import Llama3_Vision

def pil_collate_fn(batch):
    image, answer_tokens, question, image_id = zip(*batch)  # Assuming each item is a tuple (PIL.Image, label)
    return list(image), list(answer_tokens), list(question), list(image_id)

def extract_thought_answer(answer_text): 
    if "Answer:**" in answer_text:
        thought_answer = answer_text.split("Answer:**")[0] + "Answer:**"
    elif "Answer*:" in answer_text:
        thought_answer =  answer_text.split("Answer*:")[0] + "Answer*:"
    elif "Answer**:" in answer_text:
        thought_answer =  answer_text.split("Answer**:")[0] + "Answer**:"
    elif "Answer:" in answer_text:
        thought_answer =  answer_text.split("Answer:")[0] + "Answer:"
        answer_part = answer_text.split("Answer:**")[1]
    elif "**FINAL ANSWER:**" in answer_text:
        thought_answer =  answer_text.split("**FINAL ANSWER:**")[0] + "Answer*:"
    elif "FINAL ANSWER:" in answer_text:
        thought_answer = answer_text.split("FINAL ANSWER:")[0] + "Answer:"
    else:
        thought_answer =  ""

    return thought_answer
    
class ChartQADataset(Dataset):
    def __init__(self, image_dir = '/path/to/ChartQA/ChartQA Dataset/test/png', datapath = "LLaMAVision/tracing_information_flow/dataset/chartqa/chartqa_dataset_correct_answers.json") -> None:
        super().__init__() 
        self.image_dir = image_dir   
        with open(datapath, "r") as fp:
            self.datas = json.load(fp)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]
        answer_tokens = sample['answer_tokens']
        image_path = os.path.join(self.image_dir, sample['image_id'])
        image = [Image.open(image_path).convert('RGB')]
        question = sample['question']

        return image, answer_tokens, question, sample['image_id']


def parse_args():
    parser = argparse.ArgumentParser(description="CHARTQA Information Flow Eval")
    parser.add_argument(
        "--answer_path", type=str, default="LLaMAVision/results/chartqa"
    )
    parser.add_argument(
        "--data_path", type=str, default="LLaMAVision/tracing_information_flow/dataset/chartqa/chartqa_dataset_correct_answers.json"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1
    )
    parser.add_argument(
        "--start_idx", type=int, default=0
    )
    parser.add_argument(
        "--n_workers", type=int, default=4
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--block_types", nargs='+', type=str, default=['full_attention', 'question_to_last', 'image_to_last', 'image_to_question', 'last_to_last'], choices=['last_to_last', 'image_to_last', 'question_to_last', 'image_to_question', 'full_attention'], help="Blocking types to use"
    )
    parser.add_argument(
        "--k", type=int, default=9, help="Number of blocking window to use"
    )    
    parser.add_argument("--checkpoint_dir", type=str, default="/path/to/checkpoints")
    parser.add_argument("--image_dir", type=str, default="/path/to/ChartQA/ChartQA Dataset/test/png", help="Directory delle immagini")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.answer_path, exist_ok=True) 
    for k, v in args._get_kwargs():
        pad = ' '.join(['' for _ in range(25-len(k))])
        print(f"{k}:{pad} {v}", flush=True) 
    torch.cuda.set_device(0)
    torch.manual_seed(1)
    np.random.seed(1)
    
    print('Initializing Model')
    model = Llama3_Vision(args.checkpoint_dir) 
    model.eval()
    print('Initialization Finished')
    dataset = ChartQADataset(image_dir=args.image_dir, datapath=args.data_path, FA=args.FA) 
    dataloader = DataLoader(dataset, batch_size = args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True, drop_last=False, collate_fn=pil_collate_fn)

    print("Starting...")
    predictions = []
    index = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, answer_tokens, questions, image_ids = data
            prompts = []

            if index < args.start_idx:
                index += len(questions)
                continue
            
            for question in questions:
                message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {
                                "type": "text",
                                "text": (
                                    "You are provided a chart image and will be asked a question. "
                                    "You have to think through your answer and provide a step-by-step solution. "
                                    "Once you have the solution, write the final answer in at most a few words at the end with the phrase 'FINAL ANSWER:'. "
                                    f"The question is: {question}<cot_start>Let's think step by step."
                                )
                            }
                        ]
                    }
                ]
                prompts.append(message)

            for prompt in prompts:
                print(f"Prompt:{prompt}\n", flush=True)

            prob_layers = model.generate_multimodal_with_attention_blocking(prompts=prompts, answer_tokens=answer_tokens, images=images, max_gen_len=512, block_types=args.block_types, k=args.k)
        
            for idx, question in enumerate(questions):
                sample_dict = dict()
                answer_path = args.answer_path + "/" + str(idx+index) + ".json"
                for block_type in args.block_types:
                    if block_type in ['last_to_last', 'image_to_last', 'question_to_last', 'image_to_question', 'full_attention']:
                        sample_dict[block_type] = []
                        for items in prob_layers[block_type]:
                            sample_dict[block_type].append((items[0], items[1][idx].tolist()))
                    else:
                        raise NotImplementedError
                    
                sample_dict['question'] = question
                sample_dict['image_id'] = image_ids[idx]
                
                with open(answer_path, 'w') as f:
                    json.dump(sample_dict, f, indent=4)
                
            index += len(questions)
            

                

