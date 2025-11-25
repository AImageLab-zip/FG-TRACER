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
import argparse
import torch
from PIL import Image
from generation import Predictor
import requests
from io import BytesIO

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

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
        image = load_image(image_path)
        question = sample['query']
        return image, question, label, sample['imgname']


def parse_args():
    parser = argparse.ArgumentParser(description="ChartQA Eval")
    parser.add_argument(
        "--answer_path", type=str, default="LLaVA/results/chartqa.json"
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
    predictor = Predictor(model_path=args.checkpoint_dir)
    print('Initialization Finished')
    dataset = ChartQADataset(image_dir=args.image_dir, data_human_path=args.data_human_path, data_augmented_path=args.data_augmented_path) 
    dataloader = DataLoader(dataset, batch_size = args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True, drop_last=False, collate_fn=pil_collate_fn)

    print("Starting...")
    predictions = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, questions, labels, image_ids = data
            image = images[0]
            question = questions[0]
            label = labels[0]
            message = (
                    "You are provided a chart image and will be asked a question. "
                    "You have to think through your answer and provide a step-by-step solution. "
                    "Once you have the solution, write the final answer in at most a few words at the end with the phrase 'FINAL ANSWER:'. "
                    f"The question is: {question} Let's think step by step."
            )

            print(f"Prompt:{message}\n", flush=True)

            response, answer_tokens = predictor.generate(prompt=message, image=image, max_gen_tokens=1024)
            
            predictions.append({
                'image_id': image_ids[0],
                'answer': response,
                'answer_tokens': answer_tokens.tolist(),
                'gt_answer': label,
                'question': question,                
            })

    with open(args.answer_path, 'w') as f:
        json.dump(predictions, f, indent=4)

