import sys
sys.path.append('./')
import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
import torchvision.transforms as transforms
import argparse
from generation import Qwen2_5_VL_Generation
import requests
import torch
from PIL import Image

def pil_collate_fn(batch):
    image, question, label, image_id = zip(*batch)  
    return list(image), list(question), list(label), list(image_id)

class ChartQADataset(Dataset):
    def __init__(self, image_dir = '/path/to/ChartQA/ChartQA Dataset/test/png', data_human_path='/path/to/ChartQA/ChartQA Dataset/test/test_human.json', data_augmented_path='/path/to/ChartQA/ChartQA Dataset/test/test_augmented.json') -> None:
        super().__init__() 

        self.image_dir = image_dir   
        with open(data_human_path, "r") as fp:
            self.data_human = json.load(fp)
            
        with open(data_augmented_path, "r") as fp:
            self.data_augmn = json.load(fp)

        self.datas = self.data_human + self.data_augmn
    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]
        label = sample['label']
        image_path = os.path.join(self.image_dir, sample['imgname'])
        image = Image.open(image_path).convert('RGB')
        question = sample['query']
        return image, question, label, sample['imgname']


def parse_args():
    parser = argparse.ArgumentParser(description="AVQA Eval")
    parser.add_argument(
        "--answer_path", type=str, default="Qwen2.5_VL/results/chartqa.json"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1
    )
    parser.add_argument(
        "--n_workers", type=int, default=4
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument("--checkpoint_dir", type=str, default="/path/to/checkpoints")
    parser.add_argument("--image_dir", type=str, default="/path/to/ChartQA/ChartQA Dataset/test/png", help="Directory delle immagini")
    parser.add_argument("--data_human_path", type=str, default="/path/to/ChartQA/ChartQA Dataset/test/test_human.json")
    parser.add_argument("--data_augmented_path", type=str, default="/path/to/ChartQA/ChartQA Dataset/test/test_augmented.json")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.answer_path), exist_ok=True) 
    for k, v in args._get_kwargs():
        pad = ' '.join(['' for _ in range(25-len(k))])
        print(f"{k}:{pad} {v}", flush=True) 
    torch.cuda.set_device(0)
    torch.manual_seed(1)
    np.random.seed(1)
    
    print('Initializing Model')
    model = Qwen2_5_VL_Generation(args.checkpoint_dir, max_pixels=1024*1024)
    print('Initialization Finished')
    dataset = ChartQADataset(image_dir=args.image_dir, data_human_path=args.data_human_path, data_augmented_path=args.data_augmented_path) 
    dataloader = DataLoader(dataset, batch_size = args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True, drop_last=False, collate_fn=pil_collate_fn)

    print("Starting...")
    predictions = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, questions, labels, image_ids = data
            prompts = []
            for i, question in enumerate(questions):
                message = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": images[i],  
                            },
                            {
                                "type": "text",
                                "text": (
                                    "You are provided a chart image and will be asked a question. "
                                    "You have to think through your answer and provide a step-by-step solution. "
                                    "Once you have the solution, write the final answer in at most a few words at the end with the phrase 'FINAL ANSWER:'. "
                                    f"The question is: {question} Let's think step by step."
                                )
                            }
                        ]
                    }
                ]
                prompts.append(message)

            for prompt in prompts:
                print(f"Prompt:{prompt}\n", flush=True)

            results, answer_tokens = model.generate(prompts=prompts, images=images, max_gen_tokens=512)
            # (B, generated_length)
            
            for question, pred, answer_token, image_id, label in zip(questions, results, answer_tokens, image_ids, labels):
                predictions.append({
                    'image_id': image_id,
                    'answer': pred,
                    'answer_tokens': answer_token.tolist(),
                    'gt_answer': label,
                    'question': question,                
                })

    with open(args.answer_path, 'w') as f:
        json.dump(predictions, f, indent=4)

