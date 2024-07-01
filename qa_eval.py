# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/alibaba/FederatedScope/blob/dev/llm/federatedscope/llm/eval/eval_for_gsm8k/eval.py

import re
import json
import random
import torch
import numpy as np
import transformers
from tqdm import tqdm, trange
import argparse

from dola import DoLa
from collections import OrderedDict

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def load_jsonl(file_path,
               context='context',
               question='question',
               output='answer'):

    list_data_dict = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            new_item = dict(
                context=item['context'],
                question=item['question'],
                output=item['answer'])
            item = new_item
            list_data_dict.append(item)
    return list_data_dict

def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer

def build_prompt(input_text):
    context = input_text['context']
    question = input_text['question']
    label = input_text['output']
    input_text_prompt = "context: " + context + '\n' + 'question: ' + question + "?\n\n" + "answer:"
    return input_text_prompt, label

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="yanolja/EEVE-Korean-10.8B-v1.0")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--dola_layers", type=str, default="12,14,16,20,32")
    parser.add_argument("--max_gpu_memory", type=int, default=48)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="/home/blue/hyj/dola_custom/qa_dict_v2.jsonl")
    parser.add_argument("--output-path", type=str, default="result.json")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device

    # Get test fileb
    fp = args.data_path
    list_data_dict = load_jsonl(fp)

    if args.debug:
        list_data_dict = list_data_dict[:3]
    
    llm = DoLa(model_name, device, num_gpus, args.dola_layers, args.max_gpu_memory)

with open("result_dola_low_v2.jsonl", "a", encoding="utf-8") as f:
    for sample in tqdm(list_data_dict[:409]):
        result = OrderedDict()
        input_text, label = build_prompt(sample)
        model_output, origin_output_str = llm.generate(input_text)
        print('TRUE LABEL: \n{0}'.format(label))
        print('ORIGIN OUTPUT: \n{0}'.format(origin_output_str))
        print('DOLA OUTPUT: \n{0}'.format(model_output))

        result['input_text'] = input_text
        result['true_answer'] = label
        result['eeve_output'] = origin_output_str
        result['dola_output'] = model_output

        json.dump(result, f, ensure_ascii=False)
        f.write("\n")

    print("!!! End Processing !!!")