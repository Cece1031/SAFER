import argparse
import pathlib
import pickle
import tqdm
import os
import random
import json

import accelerate


import scipy.stats as st

import numpy as np
import torch
import matplotlib.pyplot as plt


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
                    default=r'Llama-3.1-8B-Instruct',
                    help='model version')
parser.add_argument('--dataset', default='sciqa')
parser.add_argument('--max-length-of-generation', type=int, default=128)
parser.add_argument('--num-beams', type=int, default=5, help='for the most likely generation')
parser.add_argument('--num-generations-per-prompt', type=int, default=20, help='for sampling')
parser.add_argument('--top-p', type=float, default=0.9, help='for sampling')
parser.add_argument('--temperature', type=float, default=1.0, help='for sampling')
# ----------------------------------------------------------------------------------------------------------------------
# for relevance method
parser.add_argument('--relevance', type=int, default=0, help='similarity, rougel, entailment, llm')
parser.add_argument('--relevance-threshold', type=float, default=0.6)
# for risk control
parser.add_argument('--split-ratio', type=float, default=0.5, help='for calib and test num')
parser.add_argument('--multi-check', type=int, default=100, help='for multiple split and check')
parser.add_argument('--delta', type=float, default=0.05, help='significance level')
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
with open(f'{run_name}/cleaned_generations.pkl', 'rb') as record_file:
    generations = pickle.load(record_file)
with open(f'{run_name}/similarity_scores.pkl', 'rb') as record_file:
    id_to_similarity_scores = pickle.load(record_file)[-1]
with open(f'{run_name}/entailment_scores.pkl', 'rb') as record_file:
    id_to_entailment_scores = pickle.load(record_file)[-1]
with open(f'{run_name}/llm_scores.pkl', 'rb') as record_file:
    id_to_llm_scores = pickle.load(record_file)[-1]
with open(f'{run_name}/rougel_scores.pkl', 'rb') as record_file:
    id_to_rougel_scores = pickle.load(record_file)[-1]
id_to_similarity = [id_to_similarity_scores, id_to_rougel_scores, id_to_entailment_scores, id_to_llm_scores][args.relevance]
# ----------------------------------------------------------------------------------------------------------------------
qa_num = len(generations)
num_calib = int(qa_num * args.split_ratio)
num_test = qa_num - num_calib
# ----------------------------------------------------------------------------------------------------------------------
emr_mean_list = []
emr_std_list = []
sr_mean_list = []
sr_std_list = []
for sampling_size in range(1, args.num_generations_per_prompt + 1):
    multi_check_emr_list = []
    multi_check_sr_list = []
    count = 0
    for epoch in range(args.multi_check):
        random.seed(epoch)
        calib_set = random.sample(generations, num_calib)
        test_set = [gen for gen in generations if gen not in calib_set]
        # empirical results on the calibration set
        miscoverage_count = 0
        for calib_data in calib_set:
            id = calib_data['id']
            relevance_score_for_correctness = id_to_similarity[id]
            miscoverage_count += int(max(relevance_score_for_correctness[:sampling_size]) < args.relevance_threshold)
        r_hat = miscoverage_count / num_calib
        # clopper pearson upper bound
        r_upper = st.beta.ppf(1 - args.delta, miscoverage_count + 1, num_calib - miscoverage_count)

        # test-time risk
        test_miscoverage_count = 0
        for test_data in test_set:
            id = test_data['id']
            relevance_score_for_correctness = id_to_similarity[id]
            test_miscoverage_count += int(max(relevance_score_for_correctness[:sampling_size]) < args.relevance_threshold)

        error_rate = test_miscoverage_count / num_test
        if error_rate <= r_upper:
            count += 1

        multi_check_sr_list.append(r_upper)
        multi_check_emr_list.append(error_rate)
    mean_sr = np.mean(multi_check_sr_list)
    mean_emr = np.mean(multi_check_emr_list)
    success_rate = count / args.multi_check
    print(multi_check_sr_list)
    print(multi_check_emr_list)
    print(mean_sr)
    print(mean_emr)
    print(success_rate)

    emr_mean_list.append(mean_emr)
    sr_mean_list.append(mean_sr)

print(list(range(1, args.num_generations_per_prompt + 1)))
print([round(item, 4) for item in sr_mean_list])
print([round(item, 4) for item in emr_mean_list])
