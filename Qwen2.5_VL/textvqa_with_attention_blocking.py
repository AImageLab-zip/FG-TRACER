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
    image, question, label, question_id = zip(*batch)  
    return list(image), list(question), list(label), list(question_id)

class TextVQADataset(Dataset):
    def __init__(self, image_dir = '/path/to/TextVQA/train_val_images/train_images', data_path = "Qwen2.5_VL/tracing_information_flow/datasets/textvqa/textvqa_dataset_correct_answers.json") -> None:
        super().__init__() 

        self.image_dir = image_dir   
        with open(data_path, "r") as fp:
            self.datas = json.load(fp)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]
        question_id = sample['questionId']
        answer_tokens = sample['answer_tokens']
        image_path = os.path.join(self.image_dir, sample['image_id']+'.jpg')
        image = Image.open(image_path).convert('RGB')
        question = sample['question']
        return image, question, answer_tokens, question_id


def parse_args():
    parser = argparse.ArgumentParser(description="AVQA Eval")
    parser.add_argument(
        "--answer_path", type=str, default="Qwen2.5_VL/results_token_masking/textvqa"
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
    parser.add_argument("--image_dir", type=str, default="/path/to/TextVQA/train_val_images/train_images", help="Directory of images")
    parser.add_argument("--data_path", type=str, default="Qwen2.5_VL/tracing_information_flow/dataset/textvqa/textvqa_dataset_correct_answers.json", help="Path to TextVQA dataset")
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
    model = Qwen2_5_VL_Generation(args.checkpoint_dir, max_pixels=1024*1024)
    model.eval()
    print('Initialization Finished')
    dataset = TextVQADataset(image_dir=args.image_dir, data_path=args.data_path) 
    dataloader = DataLoader(dataset, batch_size = args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True, drop_last=False, collate_fn=pil_collate_fn)

    print("Starting...")
    predictions = []
    index = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, questions, answer_tokens, questionIds = data
            prompts = []
            
            if index < args.start_idx:
                index += len(questions)
                continue

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
                                    "Read the text in the image carefully and answer the question with the text as seen exactly in the image. "
                                    "For yes/no questions, just respond Yes or No. "
                                    "If the answer is numeric, just respond with the number and nothing else. "
                                    "If the answer has multiple words, just respond with the words and absolutely nothing else. "
                                    f"Never respond in a sentence or a phrase.\n Question: {question}"
                                )
                            }
                        ]
                    }
                ]
                prompts.append(message)

            for prompt in prompts:
                print(f"Prompt:{prompt}\n", flush=True)

            prob_layers = model.generate_multimodal_with_attention_blocking(prompts=prompts, answer_tokens=answer_tokens, max_gen_len=124, block_types=args.block_types, k=args.k)
            # dict(block_type, list of probabilities) with probabilities = (B, n_layers)/(B)

            for idx, question_id in enumerate(questionIds):
                sample_dict = dict()
                answer_path = args.answer_path + "/" + str(question_id) + ".json"
                for block_type in args.block_types:
                    if block_type in ['last_to_last', 'image_to_last', 'question_to_last', 'image_to_question', 'full_attention', 'layers_pred']:
                        sample_dict[block_type] = []
                        for items in prob_layers[block_type]:
                            sample_dict[block_type].append((items[0], items[1][idx].tolist()))
                    else:
                        raise NotImplementedError

                with open(answer_path, 'w') as f:
                    json.dump(sample_dict, f, indent=4)


            index += len(questions)

          

