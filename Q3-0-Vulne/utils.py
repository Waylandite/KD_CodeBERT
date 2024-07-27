import os
import json
import torch
import random
import logging
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
# 知识蒸馏计算公式
def distill_loss(predicts, targets, temperature=1.0):
    predicts = predicts / temperature
    targets = targets / temperature
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return (- targets_prob * student_likelihood).mean() * temperature ** 2

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# mapFunction_convert
def mapFunction_convert(mapFunction):

    return [
        mapFunction[0],
        mapFunction[1],
        mapFunction[2],
        mapFunction[3],
        mapFunction[4],
        mapFunction[5],
    ]

def Hyperparameters_convert(hyperparameters):
    return [
        hyperparameters['learning_rate'],
        hyperparameters['hid_epoch'],
        hyperparameters['hidden_layers'],
        hyperparameters['mapfunction_1'],
        hyperparameters['mapfunction_2'],
        hyperparameters['mapfunction_3'],
        hyperparameters['mapfunction_4'],
        hyperparameters['mapfunction_5'],
        hyperparameters['mapfunction_6'],
        hyperparameters['mapfunction_7'],
        hyperparameters['mapfunction_8'],
        hyperparameters['mapfunction_9'],
        hyperparameters['mapfunction_10'],
        hyperparameters['mapfunction_11'],
        int(hyperparameters['mapfunction_12']),
    ]

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        logger.info("Creating features from file at %s ", file_path)

        data = []
        with open(file_path) as f:
            for line in f:
                data.append(json.loads(line.strip()))

        for d in tqdm(data):
            self.examples.append(convert_examples_to_features(d, tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)

class InputFeatures(object):

    def __init__(self,
                 input_tokens,
                 input_ids,
                 label
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label

def convert_examples_to_features(data, tokenizer, args):
    # get code tokens
    code = " ".join(data["func"].split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size-2]
    # add special tokens
    source_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    # convert tokens to ids
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    # add padding
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length

    return InputFeatures(source_tokens, source_ids, data["target"])


