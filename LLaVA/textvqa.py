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
    image, question, label, questionId = zip(*batch)  
    return list(image), list(question), list(label), list(questionId)

class TextVQADataset(Dataset):
    def __init__(self, image_dir = '/path/to/TextVQA/train_val_images/train_images', data_path="/path/to/TextVQA/TextVQA_0.5.1_val.json") -> None:
        super().__init__() 

        self.image_dir = image_dir   
        with open(data_path, "r") as fp:
            self.data = json.load(fp)
        self.datas = self.data['data'] #5000 questions / 3,166 images

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]

        questionId = sample['question_id']
        label = sample['answers']
        question = sample['question'] 

        image_path = os.path.join(self.image_dir, sample['image_id']+'.jpg')
        image = load_image(image_path)

        return image, question, label, questionId


def parse_args():
    parser = argparse.ArgumentParser(description="TextVQA Eval")
    parser.add_argument(
        "--answer_path", type=str, default="LLaVA/results/textvqa.json"
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
    parser.add_argument(
        "--data_path", type=str, default="/path/to/TextVQA/TextVQA_0.5.1_val.json", help="Path to TextVQA JSON file"
    )
    parser.add_argument("--checkpoint_dir", type=str, default="/path/to/checkpoints")
    parser.add_argument("--image_dir", type=str, default="/path/to/TextVQA/train_val_images/train_images", help="Directory of images")
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
    dataset = TextVQADataset(image_dir=args.image_dir, data_path=args.data_path) 
    dataloader = DataLoader(dataset, batch_size = args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True, drop_last=False, collate_fn=pil_collate_fn)

    print("Starting...")
    predictions = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, questions, labels, questionIds = data
            image = images[0]
            question = questions[0]

            response, answer_tokens = predictor.generate(prompt=question, image=image, max_gen_tokens=1024)
            # (B, generated_length)
            

            predictions.append({
                'questionId': questionIds[0],
                'answer': response, #string of generated text
                'answer_tokens': answer_tokens.tolist(), #tensor of generated ids
                'gt_answer': labels[0],
                'question': question,                
            })
            
    with open(args.answer_path, 'w') as f:
        json.dump(predictions, f, indent=4)

