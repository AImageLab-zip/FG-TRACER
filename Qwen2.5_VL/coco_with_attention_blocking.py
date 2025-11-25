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
    image, id, answer_tokens = zip(*batch)  # Assuming each item is a tuple (PIL.Image, label)
    return list(image), list(id), list(answer_tokens)

class COCODataset(Dataset):
    def __init__(self, image_dir =  '/path/to/datasets/coco/val2014', data_path = "Qwen2.5_VL/tracing_information_flow/datasets/coco/coco_dataset_correct_answers.json") -> None:
        super().__init__() 

        self.image_dir = image_dir   
        with open(data_path, "r") as fp:
            self.datas = json.load(fp)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]
        id = sample['image_id']
        answer_tokens = sample['answer_tokens']
        image_path = os.path.join(self.image_dir, sample['file_name'])
        image = Image.open(image_path).convert('RGB')
        return image, id, answer_tokens


def parse_args():
    parser = argparse.ArgumentParser(description="AVQA Eval")
    parser.add_argument(
        "--answer_path", type=str, default="Qwen2.5_VL/results_token_masking/coco"
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
        "--block_types", nargs='+', type=str, default=['question_to_last', 'image_to_last', 'image_to_question', 'last_to_last', 'full_attention'], choices=['last_to_last', 'image_to_last', 'question_to_last', 'image_to_question', 'full_attention', 'layers_pred'], help="Blocking types to use"
    )
    parser.add_argument(
        "--k", type=int, default=9, help="Number of blocking window to use"
    )
    parser.add_argument("--checkpoint_dir", type=str, default="/path/to/checkpoints")
    parser.add_argument("--image_dir", type=str, default="/path/to/datasets/coco/val2014", help="Directory of images")
    parser.add_argument("--data_path", type=str, default="Qwen2.5_VL/tracing_information_flow/datasets/coco/coco_dataset_correct_answers.json")
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
            images, ids, answer_tokens = data

            if index < args.start_idx:
                index += len(ids)
                continue

            prompts = []
            for i, _ in enumerate(ids):
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
            
            prob_layers = model.generate_multimodal_with_attention_blocking(prompts=prompts, answer_tokens=answer_tokens, max_gen_len=512, block_types=args.block_types, k=args.k)
            # dict(block_type, list of probabilities) with probabilities = (B, n_layers)/(B)

            for idx, id in enumerate(ids):
                sample_dict = dict()
                answer_path = args.answer_path + "/" + str(id) + ".json"
                for block_type in args.block_types:
                    if block_type in ['last_to_last', 'image_to_last', 'question_to_last', 'image_to_question', 'full_attention', 'layers_pred']:
                        sample_dict[block_type] = []
                        for items in prob_layers[block_type]:
                            sample_dict[block_type].append((items[0], items[1][idx].tolist()))
                    else:
                        raise NotImplementedError
                with open(answer_path, 'w') as f:
                    json.dump(sample_dict, f, indent=4)
        
            index += len(ids)
            


