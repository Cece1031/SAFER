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
                    default=r'Qwen2.5-14B-Instruct',
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
elif args.dataset in ['triviaqa', 'coqa']:
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
# load tokenizer
generative_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path,
                                                     local_files_only=True,
                                                     # resume_download=True,
                                                     # cache_dir=arg.cache_dir,
                                                     # use_auth_token="your_token",
                                                     # proxies='xxx',
                                                     # trust_remote_code=True,
                                                     use_fast=False)
# ----------------------------------------------------------------------------------------------------------------------
# load generation
with open(f'{run_name}/generations.pkl', 'rb') as record_file:
    generations = pickle.load(record_file)
# load and encode dataset
dataset = datasets.load_from_disk(f'{args.data_dir}/{args.dataset}_{model_name}')
id_to_prompt_mapping = dict(zip(dataset['id'], dataset['prompt']))  # dict{id: question, ...}

strings_to_filter_on = ['\n', '###', 'User', 'Assistant', 'System', '.']

cleaned_generations = []
for generation in tqdm.tqdm(generations):
    # for saving
    flag = True
    id = generation['id']
    prompt = id_to_prompt_mapping[id]
    sampled_generated_texts = generation['sampled_generated_texts']
    if 'most_likely_generated_text' in generation:
        most_likely_generated_text = generation['most_likely_generated_text']
        for string in strings_to_filter_on:
            if string in most_likely_generated_text:
                most_likely_generated_text = most_likely_generated_text.split(string)[0]
        if most_likely_generated_text == '' or (not most_likely_generated_text.isascii()):
            flag = False
        generation['most_likely_generated_text'] = most_likely_generated_text

    # sampled
    cleaned_generated_texts = []
    for i, generated_text in enumerate(sampled_generated_texts):
        cleaned_text = generated_text
        for string in strings_to_filter_on:
            if string in cleaned_text:
                cleaned_text = cleaned_text.split(string)[0]
        cleaned_text = cleaned_text.replace('\"', '')
        if (not cleaned_text.isascii()) or cleaned_text == '':
            flag = False
        cleaned_generated_texts.append(cleaned_text)

        # clean_ids = torch.cat(
        #     [input_ids,
        #      torch.tensor(generative_tokenizer(generated_text)['input_ids'][1:])])
        # cleaned_sampled_generations[i, :min(len(clean_ids), max_len_of_generations)] = clean_ids[:max_len_of_generations]

    generation['sampled_generated_texts'] = cleaned_generated_texts
    # generation['sampled_generation_ids'] = cleaned_sampled_generations

    if flag:
        for gen in generation['sampled_generated_texts']:
            print(gen)

        cleaned_generations.append(generation)

# save
with open(f'{run_name}/cleaned_generations.pkl', 'wb') as record_file:
    pickle.dump(cleaned_generations, record_file)
print('Record saved to ', f'{run_name}/cleaned_generations.pkl')