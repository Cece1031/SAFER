import argparse
import pathlib
import pickle
import tqdm
import os
import random
import json

import accelerate
import datasets
from collections import Counter

from transformers import AutoModelForCausalLM, AutoTokenizer

import torch.nn.functional as F
import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir',
                    default='../datasets',
                    help='save parsed dataset')
parser.add_argument('--cache-dir',
                    default='../cache',
                    help='cache model from hugging face')
parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
# ----------------------------------------------------------------------------------------------------------------------
# for run name
parser.add_argument('--record-dir',
                    default='../records',
                    help='save experimental records')
parser.add_argument('--cache-model', default=r'<PATH_TO_MODEL_CACHE>',
                    help='local path of generative llm downloaded from Hugging Face')
parser.add_argument('--generate-model',
                    default=r'openchat_3.5',
                    help='model version')
parser.add_argument('--dataset', default='sciqa')
parser.add_argument('--max-length-of-generation', type=int, default=128)
parser.add_argument('--num-beams', type=int, default=5, help='for the most likely generation')
parser.add_argument('--num-generations-per-prompt', type=int, default=20, help='for sampling')
parser.add_argument('--top-p', type=float, default=0.9, help='for sampling')
parser.add_argument('--temperature', type=float, default=1.0, help='for sampling')
# ----------------------------------------------------------------------------------------------------------------------
args = parser.parse_args()
# model_name for path of saved parsed dataset
model_name = args.generate_model
print('Generative LLM: ', model_name)
model_path = os.path.join(args.cache_model, args.generate_model)
print('Local path: ', model_path)
# mcqa setting
if args.dataset in ['commonsenseqa']:
    args.max_length_of_generation = 1
elif args.dataset in ['triviaqa', 'coqa', 'naturalqa']:
    args.max_length_of_generation = 36
elif args.dataset in ['sciqa']:
    args.max_length_of_generation = 24
# run_name for saving experimental record
run_name = os.path.join(args.record_dir,
                        args.dataset,
                        model_name,
                        'num_generations-' + str(args.num_generations_per_prompt),  # for sampling
                        'temperature-' + str(args.temperature),  # for sampling
                        'num_beams-' + str(args.num_beams),  # for most likely generation
                        'max_len_of_generation-' + str(args.max_length_of_generation))
# ----------------------------------------------------------------------------------------------------------------------
# Set seed for recurrence
seed_value = 10
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Fix torch random seed
torch.manual_seed(seed_value)
# cache path for hf_datasets
os.environ["HF_DATASETS_CACHE"] = args.cache_dir
# set cuda device 0,1
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ----------------------------------------------------------------------------------------------------------------------
# load LLM, tokenizer
generative_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path,
                                                     local_files_only=True,
                                                     # resume_download=True,
                                                     # cache_dir=arg.cache_dir,
                                                     # use_auth_token="your_token",
                                                     # proxies='xxx',
                                                     # trust_remote_code=True,
                                                     use_fast=False)
if 'Qwen' in model_name:
    dtype = torch.bfloat16
else:
    dtype = torch.float16
generative_llm = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path,
                                                      local_files_only=True,
                                                      torch_dtype=dtype,
                                                      # resume_download=True,
                                                      # cache_dir=arg.cache_dir,
                                                      # use_auth_token="your_token",
                                                      # proxies='xxx',
                                                      # trust_remote_code=True,
                                                      device_map="auto")  # require accelerate
# ----------------------------------------------------------------------------------------------------------------------
# load generation
with open(f'{run_name}/cleaned_generations.pkl', 'rb') as record_file:
    generations = pickle.load(record_file)
# load and encode dataset
dataset = datasets.load_from_disk(f'{args.data_dir}/{args.dataset}_{model_name}')
id_to_prompt_mapping = dict(zip(dataset['id'], dataset['prompt']))  # dict{id: question, ...}

def get_answer_token_entropy(model, tokenizer, prompt: str, answer: str):
    prompt_ids = tokenizer(prompt)['input_ids'][1:]
    answer_ids = tokenizer(answer)['input_ids'][1:]

    input_ids = torch.tensor([prompt_ids + answer_ids]).cuda()

    # 构造 labels，只保留 answer 部分的 loss
    labels = input_ids.clone()
    labels[0, :len(prompt_ids)] = -100  # prompt 部分忽略
    if tokenizer.pad_token_id is not None:
        labels[input_ids == tokenizer.pad_token_id] = -100  # 忽略 padding

    # 前向计算 logits
    with torch.no_grad():
        output = model(torch.reshape(input_ids, (1, -1)), labels=labels)
        logits = output.logits  # shape: [1, seq_len, vocab_size]

    # 计算 token-wise entropy
    vocab_size = logits.size(-1)
    assert model.config.vocab_size == vocab_size
    shifted_logits = logits[..., :-1, :].reshape(-1, vocab_size)
    shifted_labels = labels[..., 1:].reshape(-1)
    token_entropy = torch.nn.CrossEntropyLoss(reduction='none')(shifted_logits, shifted_labels)

    # 筛选出 answer 部分对应的 entropy
    valid_mask = shifted_labels != -100
    entropies = token_entropy[valid_mask].cpu().tolist()
    print(entropies)

    # 获取对应的 token 字符串
    full_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    answer_tokens = full_tokens[len(prompt_ids): len(prompt_ids) + len(entropies)]

    return answer_tokens, entropies


results = {}
for generation in tqdm.tqdm(generations):
    id = generation['id']
    prompt = id_to_prompt_mapping[id]
    sampled_generated_texts = generation['sampled_generated_texts']
    entropies_list = []
    for gen in sampled_generated_texts:
        _, entropies = get_answer_token_entropy(generative_llm, generative_tokenizer, prompt, gen)
        if len(entropies) != 0:
            lnpe = torch.mean(torch.tensor(entropies))
            print(lnpe)
            entropies_list.append(entropies)
    if len(entropies_list) == len(sampled_generated_texts):
        results[id] = entropies_list

# save
with open(f'{run_name}/entropies.pkl', 'wb') as record_file:
    pickle.dump(results, record_file)
print('Record saved to ', f'{run_name}/entropies.pkl')