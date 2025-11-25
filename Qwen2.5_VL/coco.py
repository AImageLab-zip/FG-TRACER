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
from generation import Qwen2_5_VL_Generation


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
        image = Image.open(image_path).convert('RGB')
        return image, id


def parse_args():
    parser = argparse.ArgumentParser(description="AVQA Eval")
    parser.add_argument(
        "--answer_path", type=str, default="Qwen2.5_VL/results/coco.json"
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
    model = Qwen2_5_VL_Generation(args.checkpoint_dir)
    model.eval()
    print('Initialization Finished')
    dataset = COCODataset(image_dir=args.image_dir, data_path=args.data_path) 
    dataloader = DataLoader(dataset, batch_size = args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True, drop_last=False, collate_fn=pil_collate_fn)

    print("Starting...")
    predictions = []
    index = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, image_ids = data

            prompts = []
            for i, _ in enumerate(image_ids):
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
                                    "Provide a detailed description for the given image in one sentence."
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

            for cap, image_id, answer_token in zip(results, image_ids, answer_tokens):
                predictions.append({
                    'image_id': image_id,
                    'caption': cap,  
                    'answer_tokens': answer_token.tolist()               
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
