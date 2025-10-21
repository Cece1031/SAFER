import argparse
import pathlib
import pickle
import tqdm
import os
import random
import json

import evaluate
import accelerate
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import CrossEncoder
from llm_eval import Qwen

import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir',
                    default='../datasets',
                    help='save parsed dataset')
parser.add_argument('--cache-dir',
                    default='../cache',
                    help='cache model from hugging face')
# ----------------------------------------------------------------------------------------------------------------------
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
# for rouge-L score
parser.add_argument('--llm', type=str, default='Qwen2.5-3B-Instruct', help='for llm evaluation')
# ----------------------------------------------------------------------------------------------------------------------
args = parser.parse_args()
# model_name for path of saved parsed dataset
model_name = args.generate_model
print('Generative LLM: ', model_name)
model_path = os.path.join(args.cache_model, args.generate_model)
print('Local path: ', model_path)
if args.dataset in ['coqa', 'triviaqa', 'naturalqa']:
    args.max_length_of_generation = 36
elif args.dataset in ['sciqa']:
    args.max_length_of_generation = 24
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
# ----------------------------------------------------------------------------------------------------------------------
# load generation
with open(f'{run_name}/cleaned_generations.pkl', 'rb') as record_file:
    generations = pickle.load(record_file)

# for llm evaluation
LLM_MAP = {
    'Qwen2.5-0.5B-Instruct': Qwen,
    'Qwen2.5-1.5B-Instruct': Qwen,
    'Qwen2.5-3B-Instruct': Qwen,
    'Qwen2.5-7B-Instruct': Qwen,
}

def obtain_llm(args):
    llm_class = LLM_MAP.get(args.llm)
    if not llm_class:
        raise ValueError(f"Unsupported LLM: {args.llm}")
    return llm_class(args.llm)

llm = obtain_llm(args)
# ----------------------------------------------------------------------------------------------------------------------
similarity_dict = dict()
similarity_for_correctness = dict()
for generation in tqdm.tqdm(generations):
    id = generation['id']
    gt_answer = generation['answer']
    sampled_generated_texts = generation['sampled_generated_texts']
    question = generation['question']
    # ------------------------------------------------------------------------------------------------------------------
    # similarity_dict[id] = dict()  # {0: [], 1: [], ...}
    # for i, gen in enumerate(sampled_generated_texts):
    #     similarity_dict[id][i] = []
    #     for j, gen_temp in enumerate(sampled_generated_texts):
    #         # qa_1 = question + ' ' + gen
    #         # qa_2 = question + ' ' + gen_temp
    #         if j == i:
    #             similarity_dict[id][i].append(1.0)
    #         elif j > i:
    #             for_check = f"Ground truth: {gen}. Model answer: {gen_temp}. Please verify if the model ans matches the ground truth. Respond with either 'Correct' or 'Wrong' only."
    #             llm_ans_check = llm.generate(
    #                 for_check,
    #                 0.1
    #             )
    #             if 'Correct' in llm_ans_check or 'correct' in llm_ans_check or 'C' in llm_ans_check or 'c' in llm_ans_check:
    #                 llm_score = 1.0
    #             else:
    #                 llm_score = 0
    #             similarity_dict[id][i].append(llm_score)
    #         elif j < i:
    #             similarity_dict[id][i].append(similarity_dict[id][j][i])
    #     print(f'id-{id} --- llm score-{i}: ', similarity_dict[id][i])
    # ------------------------------------------------------------------------------------------------------------------
    similarity_to_gt_list = []
    for sampled_gen in sampled_generated_texts:
        # qa_1 = question + ' ' + gt_answer
        # qa_2 = question + ' ' + sampled_gen
        for_check = f"Ground truth: {gt_answer}. Model answer: {sampled_gen}. Please verify if the model ans matches the ground truth. Respond with either 'Correct' or 'Wrong' only."
        llm_ans_check = llm.generate(
            for_check,
            0.1
        )
        if 'Correct' in llm_ans_check or 'correct' in llm_ans_check or 'C' in llm_ans_check or 'c' in llm_ans_check:
            llm_score = 1.0
        else:
            llm_score = 0
        similarity_to_gt_list.append(llm_score)
    similarity_for_correctness[id] = similarity_to_gt_list
    print(f'id-{id} --- llm score for correctness: ', similarity_for_correctness[id])

# similarity_dict: sampled responses之间
# similarity_for_correctness: sampled responses与gt之间
results = [similarity_dict, similarity_for_correctness]
# save
with open(f'{run_name}/llm_scores.pkl', 'wb') as record_file:
    pickle.dump(results, record_file)
print('Record saved to ', f'{run_name}/llm_scores.pkl')

