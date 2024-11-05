import os
import json
import torch
import random
import logging
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import Dataset
from tokenizers.trainers import WordLevelTrainer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors, normalizers
from transformers import RobertaTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "true"
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




def BPE(texts, vocab_size, file_path, logger):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Lowercase(),
            normalizers.NFKD(),
            normalizers.Strip(),
            normalizers.StripAccents(),
        ]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
        unk_token="<unk>"
    )

    tokenizer.train_from_iterator(texts, trainer)
    folder = "/".join(file_path.split("/")[:-1])
    tokenizer_path = os.path.join(
        folder, "BPE" + "_" + str(vocab_size) + ".json")
    tokenizer.save(tokenizer_path, pretty=True)
    logger.info("Creating vocabulary to file %s", tokenizer_path)

    return tokenizer

def WordPiece(texts, vocab_size, file_path, logger):
    tokenizer = Tokenizer(models.WordPiece())
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Lowercase(),
            normalizers.NFKD(),
            normalizers.Strip(),
            normalizers.StripAccents(),
        ]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece(prefix="##")

    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
        unk_token="<unk>"
    )

    tokenizer.train_from_iterator(texts, trainer)
    folder = "/".join(file_path.split("/")[:-1])
    tokenizer_path = os.path.join(
        folder, "WordPiece" + "_" + str(vocab_size) + ".json")
    tokenizer.save(tokenizer_path, pretty=True)
    logger.info("Creating vocabulary to file %s", tokenizer_path)

    return tokenizer

def Unigram(texts, vocab_size, file_path, logger):
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Lowercase(),
            normalizers.NFKD(),
            normalizers.Strip(),
            normalizers.StripAccents(),
        ]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder = decoders.Metaspace()

    # tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
        unk_token="<unk>"
    )

    tokenizer.train_from_iterator(texts, trainer)
    folder = "/".join(file_path.split("/")[:-1])
    tokenizer_path = os.path.join(
        folder, "Unigram" + "_" + str(vocab_size) + ".json")
    tokenizer.save(tokenizer_path, pretty=True)
    logger.info("Creating vocabulary to file %s", tokenizer_path)

    return tokenizer

def Word(texts, vocab_size, file_path, logger):
    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Lowercase(),
            normalizers.NFKD(),
            normalizers.Strip(),
            normalizers.StripAccents(),
        ]
    )

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Whitespace(),
            pre_tokenizers.Digits(individual_digits=True)
        ]
    )

    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"]
    )

    tokenizer.train_from_iterator(texts, trainer)
    folder = "/".join(file_path.split("/")[:-1])
    tokenizer_path = os.path.join(
        folder, "Word" + "_" + str(vocab_size) + ".json")
    tokenizer.save(tokenizer_path, pretty=True)
    logger.info("Creating vocabulary to file %s", tokenizer_path)

    return tokenizer


def Hyperparameters_convert(hyperparameters):
    tokenizer_type = {1: "BPE", 2: "WordPiece", 3: "Unigram", 4: "Word",5: "BERT-BPE"}
    hidden_act = {1: "gelu", 2: "relu", 3: "silu", 4: "gelu_new"}
    position_embedding_type = {1: "absolute", 2: "relative_key", 3: "relative_key_query"}
    loss_function_type = {0:"no hid_distil",1: "kl", 2: "mse" }
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


def old_hyperparams_convert(hyperparams):
    tokenizer_type = {1: "BPE", 2: "WordPiece", 3: "Unigram", 4: "Word"}
    hidden_act = {1: "gelu", 2: "relu", 3: "silu", 4: "gelu_new"}
    position_embedding_type = {1: "absolute", 2: "relative_key", 3: "relative_key_query"}
    learning_rate = {1: 1e-3, 2: 1e-4, 3: 5e-5}
    batch_size = {1: 8, 2: 16}

    return [
        tokenizer_type[hyperparams[0]],
        hyperparams[1],
        hyperparams[2],
        hyperparams[3],
        hidden_act[hyperparams[4]],
        hyperparams[5],
        hyperparams[6],
        hyperparams[7],
        hyperparams[8],
        hyperparams[9],
        position_embedding_type[hyperparams[10]],
        learning_rate[hyperparams[11]],
        batch_size[hyperparams[12]]
    ]


def ParametersCSV_convert(hyperparameters):
    tokenizer_type = {"BPE":1, "WordPiece":2,  "Unigram":3, "Word":4 ,"BERT-BPE":5}
    hidden_act = { "gelu":1,  "relu":2,  "silu":3,  "gelu_new":4}
    position_embedding_type = {"absolute":1,  "relative_key":2, "relative_key_query":3}
    loss_function_type = {"no hid_distil":0, "kl":1, "mse":2 }
    return [
        int(tokenizer_type[hyperparameters[0]]),
        int(hyperparameters[1]),
        int(hyperparameters[2]),
        int(hyperparameters[3]),    
        int(hidden_act[hyperparameters[4]]),
        hyperparameters[5],
        int(hyperparameters[6]),   
        int(hyperparameters[7]), 
        hyperparameters[8],
        int(hyperparameters[9]), 
        position_embedding_type[hyperparameters[10]],
        hyperparameters[11],
        int(hyperparameters[12]), 

        
        loss_function_type[hyperparameters[13]],
        hyperparameters[14],
        int(hyperparameters[15]),
        int(hyperparameters[16]),
        int(hyperparameters[17]),
        int(hyperparameters[18]),
        int(hyperparameters[19]),
        int(hyperparameters[20]),
        int(hyperparameters[21]),
        int(hyperparameters[22]),
        int(hyperparameters[23]),
        int(hyperparameters[24]),
        int(hyperparameters[25]),
        int(hyperparameters[26]),
        int(hyperparameters[27]),
    ]


class TextDataset(Dataset):
    def __init__(self,tokenizer_type, vocab_size, file_path, max_sequence_length ):
        self.examples = []
        logger.info("Creating features from file at %s ", file_path)
        postfix = file_path.split("/")[-1].split(".")[0]
        data = []
        with open(file_path) as f:
            for line in f:
                data.append(json.loads(line.strip()))

        folder = "/".join(file_path.split("/")[:-1])
        tokenizer_path = os.path.join(
            folder, tokenizer_type + "_" + str(vocab_size) + ".json")

        if os.path.exists(tokenizer_path):
            tokenizer = Tokenizer.from_file(tokenizer_path)
            logger.info("Loading vocabulary from file %s", tokenizer_path)
        else:
            texts = [" ".join(d["func"].split()) for d in data]
            if tokenizer_type == "BPE":
                tokenizer = BPE(texts, vocab_size, file_path, logger)
            elif tokenizer_type == "WordPiece":
                tokenizer = WordPiece(texts, vocab_size, file_path, logger)
            elif tokenizer_type == "Unigram":
                tokenizer = Unigram(texts, vocab_size, file_path, logger)
            elif tokenizer_type == "Word":
                tokenizer = Word(texts, vocab_size, file_path, logger)
            elif tokenizer_type =="BERT-BPE":
                # prepare the tokenizer
                tokenizer = RobertaTokenizer.from_pretrained("/home/wuruifeng/data/models/roberta-base")
                tokenizer.do_lower_case = True

        for d in tqdm(data):
            self.examples.append(convert_examples_to_features(d, tokenizer_type,tokenizer, max_sequence_length, postfix ))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label), torch.tensor(self.examples[i].soft_label)

class InputFeatures(object):

    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 soft_label=[0.1, 0.1]
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label
        self.soft_label = soft_label

def convert_examples_to_features(data,  tokenizer_type,tokenizer, max_sequence_length, postfix ):
    if tokenizer_type == "BERT-BPE":
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
    else:
        code = " ".join(data["func"].split())
        source_tokens= code
        source_ids = tokenizer.encode(source_tokens).ids[:max_sequence_length]
        padding_length = max_sequence_length - len(source_ids)
        source_ids += [tokenizer.token_to_id("<pad>")] * padding_length
    if "train" in postfix:
        return InputFeatures(source_tokens, source_ids, data["target"],data["soft_label"])
    else:
        return InputFeatures(source_tokens, source_ids, data["target"])




