import argparse
import pathlib
import pickle
import tqdm
import os
import random
import json


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
parser.add_argument('--dataset', default='triviaqa')
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
parser.add_argument('--split-ratio', type=float, default=0.1, help='for calib and test num')
parser.add_argument('--multi-check', type=int, default=100, help='for multiple split and check')
parser.add_argument('--alpha', type=float, default=0.05, help='risk level')
parser.add_argument('--beta', type=list, default=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5], help='risk level')
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
with open(f'{run_name}/entropies.pkl', 'rb') as record_file:
    id_to_entropies = pickle.load(record_file)
id_to_similarity = [id_to_similarity_scores, id_to_rougel_scores, id_to_entailment_scores, id_to_llm_scores][args.relevance]
# ----------------------------------------------------------------------------------------------------------------------
temp = []
for gen in tqdm.tqdm(generations):
    id = gen['id']
    if id in id_to_entropies:
        temp.append(gen)
generations = temp

# uncertainty
id_to_uncertainty = {}
for gen in tqdm.tqdm(generations):
    id = gen['id']
    entropies = id_to_entropies[id]
    # uncertainty
    lnpe_list = []
    for ent in entropies:
        lnpe = torch.tensor(ent).mean()
        lnpe_list.append(lnpe.item())
    id_to_uncertainty[id] = lnpe_list
# ----------------------------------------------------------------------------------------------------------------------
qa_num = len(generations)
num_calib = int(qa_num * args.split_ratio)
num_test = qa_num - num_calib
# ----------------------------------------------------------------------------------------------------------------------
mean_list = []
std_list = []
beta_list = args.beta
for beta in beta_list:
    multi_check_emr_list = []
    for epoch in range(args.multi_check):
        random.seed(epoch)
        calib_set = random.sample(generations, num_calib)
        test_set = [gen for gen in generations if gen not in calib_set]
        # Step 1: Find smallest s such that upper bound <= alpha (calculation of s_max)
        optimal_s = None
        for s in range(1, args.num_generations_per_prompt + 1):
            miscoverage_count = 0
            for calib_data in calib_set:
                id = calib_data['id']
                relevance_score_for_correctness = id_to_similarity[id]
                miscoverage_count += int(max(relevance_score_for_correctness[:s]) < args.relevance_threshold)
            r_hat = miscoverage_count / num_calib
            r_upper = st.beta.ppf(1 - args.delta, miscoverage_count + 1, num_calib - miscoverage_count)
            if r_upper <= args.alpha:
                optimal_s = s
                break
        if optimal_s is not None:
        # assert optimal_s is not None
            print('Minimum sampling size: ', optimal_s)
            # clean calibration set
            cleaned_calibration_set = []
            for calib_data in calib_set:
                id = calib_data['id']
                relevance_score_for_correctness = id_to_similarity[id]
                if max(relevance_score_for_correctness[:optimal_s]) >= args.relevance_threshold:
                    cleaned_calibration_set.append(calib_data)
            # Step 2: Find uncertainty threshold
            min_uncertainty = min([min(id_to_uncertainty[d['id']]) for d in cleaned_calibration_set])
            max_uncertainty = max([max(id_to_uncertainty[d['id']]) for d in cleaned_calibration_set])
            sorted_uncertainties = np.linspace(min_uncertainty, max_uncertainty, num_calib).tolist()[::-1]
            low = 0
            high = len(sorted_uncertainties) - 1
            t_hat = None
            while low <= high:
                mid = (low + high) // 2
                t_candidate = sorted_uncertainties[mid]
                l_cal = 0
                for calib_data in cleaned_calibration_set:
                    id = calib_data['id']
                    uncertainty = np.array(id_to_uncertainty[id][:optimal_s])
                    relevance_score_for_correctness = np.array(id_to_similarity_scores[id][:optimal_s])
                    if len(relevance_score_for_correctness[uncertainty <= t_candidate].tolist()) == 0:
                        l_cal += 1
                    elif np.max(relevance_score_for_correctness[uncertainty <= t_candidate]) < args.relevance_threshold:
                        l_cal += 1
                if (l_cal + 1)/(len(cleaned_calibration_set) + 1) <= beta:
                    t_hat = t_candidate
                    low = mid + 1  # 尝试更大的 t
                else:
                    high = mid - 1
            if t_hat is None:
                print('The risk level is unmanageable!')
                continue
            else:
                print(f'Min threshold t_hat = {t_hat}')
                l_test = 0
                for test_data in test_set:
                    id = test_data['id']
                    uncertainty = np.array(id_to_uncertainty[id][:optimal_s])
                    relevance_score_for_correctness = np.array(id_to_similarity_scores[id][:optimal_s])
                    if len(relevance_score_for_correctness[uncertainty <= t_hat].tolist()) == 0:
                        l_test += 1
                    elif max(relevance_score_for_correctness[uncertainty <= t_hat]) < args.relevance_threshold:
                        l_test += 1
                risk = l_test / len(test_set)
                print(risk)
                multi_check_emr_list.append(risk)
    multi_check_emr_numpy = np.array(multi_check_emr_list)
    mean = np.mean(multi_check_emr_numpy)
    std = np.std(multi_check_emr_numpy)
    mean_list.append(mean)
    std_list.append(std)
    print('-' * 20)
    print(mean)
    print(std)
    print('-' * 20)

print([args.alpha + beta - args.alpha * beta for beta in beta_list])
print(mean_list)
print(std_list)

