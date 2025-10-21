import argparse
import pathlib
import pickle
import tqdm
import os
import random
import json

import accelerate
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder

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
# for bi-entailment
parser.add_argument('--infer-model',
                    default=r'<PATH_TO_MODEL_CACHE>\deberta-large-mnli',
                    help='local path of infer llm')
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

# for entailment inference
infer_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.infer_model)
infer_llm = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.infer_model).cuda()
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
    #         qa_1 = question + ' ' + gen
    #         qa_2 = question + ' ' + gen_temp
    #         if j == i:
    #             similarity_dict[id][i].append(1.0)
    #         elif j > i:
    #             input_1 = qa_1 + ' [SEP] ' + qa_2
    #             input_2 = qa_2 + ' [SEP] ' + qa_1
    #             inputs = [input_1, input_2]
    #             encoded_input = infer_tokenizer.batch_encode_plus(inputs, padding=True)
    #             prediction = infer_llm(torch.tensor(encoded_input['input_ids']).cuda())['logits']
    #             prob_vecter = torch.softmax(prediction, dim=1).detach().to('cpu')
    #             entail_score = (prob_vecter[0, 2].item() + prob_vecter[1, 2].item()) / 2
    #             similarity_dict[id][i].append(entail_score)  # [label, prediction]
    #         elif j < i:
    #             similarity_dict[id][i].append(similarity_dict[id][j][i])
    #     print(f'id-{id} --- entailment_scores-{i}: ', similarity_dict[id][i])
    # ------------------------------------------------------------------------------------------------------------------
    similarity_to_gt_list = []
    for sampled_gen in sampled_generated_texts:
        qa_1 = question + ' ' + gt_answer
        qa_2 = question + ' ' + sampled_gen
        input_1 = qa_1 + ' [SEP] ' + qa_2
        input_2 = qa_2 + ' [SEP] ' + qa_1
        inputs = [input_1, input_2]
        encoded_input = infer_tokenizer.batch_encode_plus(inputs, padding=True)
        prediction = infer_llm(torch.tensor(encoded_input['input_ids']).cuda())['logits']
        prob_vecter = torch.softmax(prediction, dim=1).detach().to('cpu')
        entail_score = (prob_vecter[0, 2].item() + prob_vecter[1, 2].item()) / 2
        similarity_to_gt_list.append(entail_score)  # [label, prediction]
    similarity_for_correctness[id] = similarity_to_gt_list
    print(f'id-{id} --- entailment_scores for correctness: ', similarity_for_correctness[id])

# similarity_dict: sampled responses之间
# similarity_for_correctness: sampled responses与gt之间
results = [similarity_dict, similarity_for_correctness]
# save
with open(f'{run_name}/entailment_scores.pkl', 'wb') as record_file:
    pickle.dump(results, record_file)
print('Record saved to ', f'{run_name}/entailment_scores.pkl')

