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
# -------------------------------------zai---------------------------------------------------------------------------------
# for run name
parser.add_argument('--record-dir',
                    default='../records',
                    help='save experimental records')
parser.add_argument('--cache-model', default=r'<PATH_TO_MODEL_CACHE>',
                    help='local path of generative llm downloaded from Hugging Face')
parser.add_argument('--generate-model',
                    default=r'Qwen2-7B-Instruct',
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
# load and encode dataset
dataset = datasets.load_from_disk(f'{args.data_dir}/{args.dataset}_{model_name}')
id_to_prompt_mapping = dict(zip(dataset['id'], dataset['prompt']))  # dict{id: question, ...}
# split
if args.fraction_of_data_to_use < 1.0:
    dataset = dataset.train_test_split(test_size=args.fraction_of_data_to_use, seed=seed_value)['test']
print(dataset)
# encode prompt
def encode(examples):
    return generative_tokenizer(examples['prompt'], truncation=False, padding=False)

def encode_and_format_dataset(dataset):
    dataset = dataset.map(encode, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)
    return dataset

questions = encode_and_format_dataset(dataset)
# batch for input
dataloader = torch.utils.data.DataLoader(questions, batch_size=1)
# ----------------------------------------------------------------------------------------------------------------------
# special tokens
eos_token_id = generative_tokenizer('. ')['input_ids'][1]
# eos_token_id (`Union[int, List[int]]`, *optional*):
#             The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
bad_words = ['### User:', '### Assistant:', '### System:', 'User:', 'Assistant:', 'System:', '###', '\n']
# bad_words_ids = [generative_tokenizer(bad_word, add_special_tokens=False)['input_ids'] for bad_word in bad_words]
bad_words_ids = [generative_tokenizer.encode(bad_word, add_special_tokens=False) for bad_word in bad_words]
# bad_words_ids = [[generative_tokenizer(bad_word)['input_ids'][1]] for bad_word in bad_words]
# bad_words_ids(`List[List[int]]`, *optional*):
#             List of token ids that are not allowed to be generated.
# ----------------------------------------------------------------------------------------------------------------------
# get generations
def get_generations(dataloader):
    for_json_saving = []
    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
            input_ids = batch['input_ids'].cuda()
            input_length = input_ids.shape[1]
            attention_mask = batch['attention_mask'].cuda()
            assert input_ids.shape == attention_mask.shape
            # ----------------------------------------------------------------------------------------------------------
            # generate
            # # the most likely generation - for model output
            # most_likely_generation = generative_llm.generate(input_ids,
            #                                                  attention_mask=attention_mask,
            #                                                  num_beams=args.num_beams,
            #                                                  do_sample=False,
            #                                                  max_length=input_length + args.max_length_of_generation,
            #                                                  eos_token_id=eos_token_id,
            #                                                  bad_words_ids=bad_words_ids,
            #                                                  temperature=0.1)
            # sampled generations - for uncertainty quantification
            sampled_generations = torch.ones(
                (args.num_generations_per_prompt, input_length + args.max_length_of_generation),
                dtype=torch.long).cuda()
            for i in range(args.num_generations_per_prompt):
                generation = generative_llm.generate(input_ids,
                                                     attention_mask=attention_mask,
                                                     do_sample=True,
                                                     num_return_sequences=1,  # <= num_beams
                                                     num_beams=1,  # greedy search
                                                     max_length=input_length + args.max_length_of_generation,
                                                     eos_token_id=eos_token_id,
                                                     bad_words_ids=bad_words_ids,
                                                     temperature=args.temperature,
                                                     top_p=args.top_p)
                sampled_generations[i, :generation.shape[1]] = generation
            sampled_generations = torch.reshape(sampled_generations,
                                                (-1, args.num_generations_per_prompt, sampled_generations.shape[-1]))
            # ----------------------------------------------------------------------------------------------------------
            # for save
            for i in range(sampled_generations.shape[0]):  # batch_num
                generation_dict = {}
                json_save_dict = {}
                # mcqa options
                if args.dataset in ['commonsenseqa']:
                    generation_dict['options'] = [temp[i] for temp in batch['options']]
                    json_save_dict['options'] = [temp[i] for temp in batch['options']]
                # save id, question, answer
                generation_dict['id'] = batch['id'][i]
                json_save_dict['id'] = batch['id'][i]
                generation_dict['question'] = batch['question'][i]
                json_save_dict['question'] = batch['question'][i]
                generation_dict['answer'] = batch['answer'][i]
                json_save_dict['answer'] = batch['answer'][i]
                # decode and save most_likely_generated_text and sampled_generated_texts
                # most_likely_generated_text = generative_tokenizer.decode(most_likely_generation[i][input_length:],
                #                                                           skip_special_tokens=True)
                # generation_dict['most_likely_generated_text'] = most_likely_generated_text
                # json_save_dict['most_likely_generated_text'] = most_likely_generated_text
                sampled_generated_texts = []
                for generation in sampled_generations[i]:
                    sampled_generated_texts.append(
                        generative_tokenizer.decode(generation[input_length:], skip_special_tokens=True)
                    )
                generation_dict['sampled_generated_texts'] = sampled_generated_texts
                json_save_dict['sampled_generated_texts'] = sampled_generated_texts
                # display question and ground-truth answer
                print('Question: ', generation_dict['question'])
                if args.dataset in ['commonsenseqa']:
                    print('Options: ', generation_dict['options'])  # MCQA options
                print('Ground-truth Answer: ', generation_dict['answer'])
                # display generation: sampled amd the most likely
                # print('The most likely generation: ', generation_dict['most_likely_generated_text'])
                print('Sampled Generations:')
                for j in range(args.num_generations_per_prompt):
                    print([generation_dict['sampled_generated_texts'][j]])

                results.append(generation_dict)
                for_json_saving.append(json_save_dict)
    return results, for_json_saving

generations, for_json_saving = get_generations(dataloader)

# save to records
pathlib.Path(run_name).mkdir(parents=True, exist_ok=True)
with open(f'{run_name}/generations.pkl', 'wb') as record_file:
    pickle.dump(generations, record_file)
print('Record saved to ', f'{run_name}/generations.pkl')

# only for read
with open(f'{run_name}/generations.json', 'w') as json_file:
    json.dump(for_json_saving, json_file, indent=4)