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
    image, image_id = zip(*batch)  
    return list(image), list(image_id)

class COCODataset(Dataset):
    def __init__(self, image_dir = '/path/to/datasets/coco/val2014', data_path = "/path/to/datasets/coco/annotations/captions_val2014.json") -> None:
        super().__init__() 

        self.image_dir = image_dir   
        with open(data_path, "r") as fp:
            self.data = json.load(fp)
        self.datas = self.data['images']            #40504 samples
    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]
        id = sample['id']
        image_path = os.path.join(self.image_dir, sample['file_name'])
        image = load_image(image_path)
        return image, id


def parse_args():
    parser = argparse.ArgumentParser(description="COCO Eval")
    parser.add_argument(
        "--answer_path", type=str, default="LLaVA/results/coco.json"
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
    parser.add_argument("--image_dir", type=str, default="/path/to/datasets/coco/val2014", help="Directory of images")
    parser.add_argument("--data_path", type=str, default="/path/to/datasets/coco/annotations/captions_val2014.json")
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
    dataset = COCODataset(image_dir=args.image_dir, data_path=args.data_path) 
    dataloader = DataLoader(dataset, batch_size = args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True, drop_last=False, collate_fn=pil_collate_fn)

    print("Starting...")
    predictions = []
    index = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, image_ids = data
            image = images[0]

            message = "Provide a detailed description for the given image in one sentence."                
    
            print(f"Prompt:{message}\n", flush=True)

            response, answer_tokens = predictor.generate(prompt=message, image=image, max_gen_tokens=1024)
            predictions.append({
                'image_id': image_ids[0],
                'caption': response,  
                'answer_tokens': answer_tokens.tolist()               
            })

            index += len(image_ids)
            if index % 5000 == 0:
                print(f"Processed {index} images")
                output_path = f"{args.answer_path}_{index}.json"
                with open(output_path, 'w') as f:
                    json.dump(predictions, f, indent=4)

        output_path = f"{args.answer_path}_{index}.json"
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=4)
