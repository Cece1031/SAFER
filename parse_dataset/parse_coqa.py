import argparse
import json
import random

import numpy as np
import tqdm
import re
import os

import accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import pandas as pd
import datasets
from datasets import Dataset

# ----------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--cache-model', default=r'<PATH_TO_MODEL_CACHE>',
                    help='local path of generative llm downloaded from Hugging Face')
parser.add_argument('--generate-model',
                    default=r'Qwen2-7B-Instruct',
                    help='model version')
parser.add_argument('--row-data-path',
                    default=r'../row_data/coqa',
                    help='local path of row dataset')
parser.add_argument('--data-dir',
                    default='../datasets',
                    help='save parsed dataset')
parser.add_argument('--cache-dir',
                    default='../cache',
                    help='cache model from hugging face')
parser.add_argument('--few-shot-num',
                    default=1,
                    help='for few-shot prompt')
parser.add_argument('--max-num',
                    default=2000,
                    help='for save')
args = parser.parse_args()
# ----------------------------------------------------------------------------------------------------------------------
# model_name for path of saved parsed dataset
model_name = args.generate_model
print('Generative LLM: ', model_name)
model_path = os.path.join(args.cache_model, args.generate_model)
print('Local path: ', model_path)
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
# set cuda device 0,1
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
os.environ["HF_DATASETS_CACHE"] = args.cache_dir
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ----------------------------------------------------------------------------------------------------------------------
# for input_ids length (allowed by llm)
generative_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path,
                                                     local_files_only=True,
                                                     # resume_download=True,
                                                     # cache_dir=arg.cache_dir,
                                                     # use_auth_token="your_token",
                                                     # proxies='xxx',
                                                     # trust_remote_code=True,
                                                     use_fast=False)
generative_llm = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path,
                                                      local_files_only=True,
                                                      torch_dtype=torch.float16,
                                                      # resume_download=True,
                                                      # cache_dir=arg.cache_dir,
                                                      # use_auth_token="your_token",
                                                      # proxies='xxx',
                                                      # trust_remote_code=True,
                                                      device_map="auto")  # require accelerate
max_input_ids_length = generative_llm.config.max_position_embeddings
print('LLM max input ids length: ', max_input_ids_length)
# ----------------------------------------------------------------------------------------------------------------------
# load row data (download from hugging face https://huggingface.co/datasets/stanfordnlp/coqa/tree/main/data)
validation_name = 'validation-00000-of-00001.parquet'
row_data_path = os.path.join(args.row_data_path, validation_name)
print('Data path: ', row_data_path)
df = pd.read_parquet(row_data_path)
dict_list = df.to_dict(orient='records')
print('Num samples: ', len(dict_list))  # 500
num_qa = 0
for sample in dict_list:
    num_qa += len(sample['questions'])
print('Num question-answer pairs: ', num_qa)  # 7983
# 'story'    : str
# 'questions': list[str, ...]
# 'answer'   : dict{'input_text'  : array([str, ...]),
#                   'answer_start': array([int, ...]),
#                   'answer_end'  : array([int, ...])}
# ----------------------------------------------------------------------------------------------------------------------
# prompt engineer
instruction = "### System:\nThis is a bot that correctly answers questions.\n"

dataset = {}

dataset['prompt'] = []
dataset['question'] = []
dataset['answer'] = []
dataset['id'] = []
# ----------------------------------------------------------------------------------------------------------------------
# parse
applied_qa_pairs = 0
for sample_idx, sample in enumerate(tqdm.tqdm(dict_list)):
    story = sample['story']
    questions = sample['questions']
    answers = sample['answers']['input_text']
    assert len(questions) == len(answers)
    # examine story
    story = story.replace('\n', ' ')  # for line break
    story = re.sub(r"\s+", " ", story.strip())  # for multiple consecutive spaces
    story += '\n'

    few_shot_num = 0
    for idx, question in enumerate(questions):
        # examine input_ids length
        prompt = instruction + story + '\n### User:\n' + question + '\n### Assistant:\n'
        input_ids = generative_tokenizer.encode(prompt)
        if len(input_ids) < max_input_ids_length:
            applied_qa_pairs += 1

            dataset['prompt'].append(prompt)  # in context
            if applied_qa_pairs == 2:
                print(prompt)
                exit()
            dataset['question'].append(question)
            dataset['answer'].append(answers[idx])
            dataset['id'].append(str(sample_idx) + '_' + str(idx))
            # few-shot
            if few_shot_num <= args.few_shot_num:
                story += '\n### User:\n' + question + '\n### Assistant:\n' + answers[idx] + '\n'
                few_shot_num += 1

            if applied_qa_pairs == args.max_num:
                break
    if applied_qa_pairs == args.max_num:
        break
print('Applied question-answer pairs: ', applied_qa_pairs)
# ----------------------------------------------------------------------------------------------------------------------
# save
dataset_df = pd.DataFrame.from_dict(dataset)
dataset = Dataset.from_pandas(dataset_df)
dataset.save_to_disk(f'{args.data_dir}/coqa_{model_name}')
# ----------------------------------------------------------------------------------------------------------------------
# check answer length
len_list = []
for answer in dataset['answer']:
    answer_ids = generative_tokenizer.encode(answer)
    len_list.append(len(answer_ids))

print('Max answer length: ', max(len_list))
