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




def Hyperparameters_convert(hyperparameters):
    tokenizer_type = {1: "BPE", 2: "WordPiece", 3: "Unigram", 4: "Word"}
    hidden_act = {1: "gelu", 2: "relu", 3: "silu", 4: "gelu_new"}
    position_embedding_type = {1: "absolute", 2: "relative_key", 3: "relative_key_query"}
    loss_function_type = {1: "kl", 2: "mse", }   
    return [
        tokenizer_type[hyperparameters['tokenizer']],
        int(hyperparameters['vocab_size']),
        int(hyperparameters['num_hidden_layers']),
        int(hyperparameters['hidden_size']),    
        hidden_act[hyperparameters['hidden_act']],
        hyperparameters['hidden_dropout_prob'],
        int(hyperparameters['intermediate_size']),   
        int(hyperparameters['num_attention_heads']), 
        hyperparameters['attention_probs_dropout_prob'],
        int(hyperparameters['max_sequence_length']), 
        position_embedding_type[hyperparameters['position_embedding_type']],
        hyperparameters['pred_learning_rate'],
        int(hyperparameters['batch_size']), 

        
        loss_function_type[hyperparameters['loss_function']],
        hyperparameters['hid_learning_rate'],
        int(hyperparameters['hid_epoches']),
        int(hyperparameters['mapfunction_1']),
        int(hyperparameters['mapfunction_2']),
        int(hyperparameters['mapfunction_3']),
        int(hyperparameters['mapfunction_4']),
        int(hyperparameters['mapfunction_5']),
        int(hyperparameters['mapfunction_6']),
        int(hyperparameters['mapfunction_7']),
        int(hyperparameters['mapfunction_8']),
        int(hyperparameters['mapfunction_9']),
        int(hyperparameters['mapfunction_10']),
        int(hyperparameters['mapfunction_11']),
        int(hyperparameters['mapfunction_12']),
    ]

class TextDataset(Dataset):
    def __init__(self, tokenizer, max_sequence_length, file_path=None):
        self.examples = []
        logger.info("Creating features from file at %s ", file_path)

        data = []
        with open(file_path) as f:
            for line in f:
                data.append(json.loads(line.strip()))

        for d in tqdm(data):
            self.examples.append(convert_examples_to_features(d, tokenizer, max_sequence_length))

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

def convert_examples_to_features(data, tokenizer, max_sequence_length):
    # get code tokens
    code = " ".join(data["func"].split())
    code_tokens = tokenizer.tokenize(code)[:max_sequence_length-2]
    # add special tokens
    source_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    # convert tokens to ids
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    # add padding
    padding_length = max_sequence_length - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length

    return InputFeatures(source_tokens, source_ids, data["target"])



