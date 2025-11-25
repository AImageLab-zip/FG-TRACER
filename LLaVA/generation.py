
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images
from typing import Callable, Generator, List, Optional
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, model_path = "", model_name = "llava-v1.5-13b", conv_mode = "llava_v1"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Loading LLaVA')
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_name=model_name, model_base=None, load_8bit=False, load_4bit=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.conv_mode = conv_mode

        for name, param in self.model.named_parameters():
            param.requires_grad = False
        print('Loading LLaVA Done')

    @torch.inference_mode()
    def generate(
        self,
        prompt,
        image,
        max_gen_tokens = 2048,
    ):
        conv = conv_templates[self.conv_mode].copy()
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        input_text = conv.get_prompt()
        print(f"Input text: {input_text}")
        
        input_ids = tokenizer_image_token(input_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        #(1, N)
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        images = image_tensor.unsqueeze(0).half().to(self.device)
        #(1, 3, 336, 336)
        image_sizes = [image.size]

        output = self.model.generate(
                inputs=input_ids,
                images=images,
                image_sizes=image_sizes,
                max_new_tokens=max_gen_tokens,
                do_sample=False,
        )[0]
  
        output_text = self.tokenizer.decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
            add_special_tokens=False
        )
        response = output_text.strip() 
        print(f"Response: {response}\n")

        return response, output
    
    @torch.inference_mode()
    def generate_multimodal_with_attention_blocking(
        self,
        prompt,
        answer_tokens,
        image,
        thought_answer=None,
        temperature: float = 0.2,
        max_gen_len: Optional[int] = None,
        block_types: List[str] = None,
        k: int = 9,
    ): 
        
        conv = conv_templates[self.conv_mode].copy()
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt 
        if thought_answer is not None:
            inp += thought_answer
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        input_text = conv.get_prompt()
        print(f"Input text: {input_text}")

        input_ids = tokenizer_image_token(input_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        #(1, seq_length)
        params = self.model.config
        if max_gen_len is None or max_gen_len == 0 or max_gen_len >= params.max_position_embeddings:
            max_gen_len = params.max_position_embeddings - 1

        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        images = image_tensor.unsqueeze(0).half().to(self.device)
        # images = (1, 3, 336, 336)
        image_sizes = [image.size] 
        # image_sizes = [(640, 480)]
        
        answer_tokens = torch.tensor(answer_tokens, device=self.device).unsqueeze(0)  
        # answer_tokens = (1, 1, seq_length)
        prompt_len = input_ids.shape[1]
        bsz = input_ids.shape[0]

        if prompt_len >= params.max_position_embeddings:
            print(f"Out of token budget {prompt_len} vs {params.max_position_embeddings}", "red")
            return
        total_len = min(max_gen_len + prompt_len, params.max_position_embeddings)

        pad_id = self.tokenizer.pad_token_id #0
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=self.device)                
        
        for j, t in enumerate(input_ids):
            tokens[j, : len(t)] = t.clone().detach()  

        prob_layers_type = {}
        if block_types is not None:
            for block_type in block_types:
                prev_pos = 0
                cur_pos = prompt_len

                if block_type in ['last_to_last', 'image_to_last', 'question_to_last', 'image_to_question', 'full_attention', 'layers_pred']:
                    prob_layers_type[block_type] = []  # list of tuples (word, prob)
                    for i in range(answer_tokens.shape[2]):
                        answer_token = answer_tokens[:, :, i].unsqueeze(-1)                 #(1, batch_size, 1)
                        a_token = answer_token.item()
                        if a_token == self.tokenizer.eos_token_id or a_token == self.tokenizer.bos_token_id: #2 = <|eot_id|>
                            continue
                        logits = self.model.forward_with_attention_blocking(
                            input_ids=tokens[:, prev_pos:cur_pos],                          #(batch_size, seq_len)
                            images=images,                                                  #(batch_size, 3, 336, 336)
                            image_sizes=image_sizes,
                            block_type=block_type,
                            k=k,
                        )                                                                   #(n_layers, batch_size, vocab_size)/(1, batch_size, vocab_size)

                        if block_type in ['last_to_last', 'image_to_last', 'question_to_last', 'image_to_question', 'layers_pred']:
                            probs = torch.softmax(logits / temperature if temperature > 0 else logits, dim=-1)  # (n_layers, batch_size, vocab_size)
                            indices = answer_token.expand_as(logits[..., :answer_token.shape[2]])               # (n_layers, batch_size, 1)
                            new_prob = probs.gather(dim=2, index=indices).squeeze(-1).transpose(0, 1)           # (n_layers, batch_size, 1) -> (batch_size, n_layers)
                            word = self.tokenizer.decode([a_token])
                            prob_layers_type[block_type].append((word, new_prob))
                        
                        elif block_type == 'full_attention':
                            probs = torch.softmax(logits.squeeze(0) / temperature if temperature > 0 else logits, dim=-1)   # (batch_size, vocab_size)
                            indices = answer_token.squeeze(0)                                                               # (batch_size, 1)
                            new_prob = probs.gather(dim=1, index=indices).squeeze(-1)                                       # (batch_size, 1) -> (batch_size)
                            word = self.tokenizer.decode([a_token])
                            prob_layers_type[block_type].append((word, new_prob))
                            
                        else:
                            raise NotImplementedError
                        
                        tokens[:, cur_pos] = answer_tokens[0, :, i]                                                         #(batch_size, tot_len)
                        cur_pos += 1
                else:
                    raise NotImplementedError
                
        print(list(prob_layers_type.keys()))
        return prob_layers_type 
   
def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token