import argparse
import json
import random
import time
import pandas as pd
import numpy as np
import torch
from hyperopt import hp, space_eval
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
from models import Model
import csv
import logging
from utils import Hyperparameters_convert, ParametersCSV_convert
from transformers import RobertaTokenizer
from surrogate import predictor
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaModel, \
    RobertaForSequenceClassification, RobertaTokenizer
from distill_utils import distill

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)

# def check_model_size(sampled_params):
#     tokenizer_type, vocab_size, num_hidden_layers, hidden_size, hidden_act, hidden_dropout_prob, intermediate_size, num_attention_heads, attention_probs_dropout_prob, max_sequence_length, position_embedding_type, pred_learning_rate, batch_size,loss_function,hid_learning_rate,hid_epoches,*rest = Hyperparameters_convert(sampled_params)
#     # student model
#     student_config = RobertaConfig.from_pretrained("/home/wuruifeng/data/models/6layer_config")
#     student_config.num_labels = 2
#     # student_config.vocab_size = vocab_size
#     student_config.num_hidden_layers = num_hidden_layers
#     student_config.hidden_size = hidden_size
#     student_config.hidden_act = hidden_act
#     student_config.hidden_dropout_prob = hidden_dropout_prob
#     student_config.intermediate_size = intermediate_size
#     student_config.num_attention_heads = num_attention_heads
#     student_config.attention_probs_dropout_prob = attention_probs_dropout_prob
#     student_config.max_position_embeddings = max_sequence_length+2
#     student_config.position_embedding_type = position_embedding_type
#     student_model = Model(RobertaForSequenceClassification(student_config))
#     total_params = sum(p.numel() for p in student_model.parameters())
#     print(total_params * 4 / 1e6)
#     return (total_params * 4 / 1e6)<=3

# def check_model_size(sampled_params):
#     tokenizer_type, vocab_size, num_hidden_layers, hidden_size, hidden_act, hidden_dropout_prob, intermediate_size, num_attention_heads, attention_probs_dropout_prob, max_sequence_length, position_embedding_type, pred_learning_rate, batch_size,loss_function,hid_learning_rate,hid_epoches,*rest = Hyperparameters_convert(sampled_params)
#     # student model
#     embedding=4*(vocab_size+max_sequence_length+3)*hidden_size
#     transformerlayer=4*(4*hidden_size**hidden_size+(9+2*intermediate_size)*hidden_size+intermediate_size)*num_hidden_layers
#     classiter=2*hidden_size**hidden_size+4*hidden_size+2
#
#
#     total_params =embedding+transformerlayer+classiter
#     print(total_params / 1e6)
#     return (total_params / 1e6)<=3

# 判断是否存在学习冲突问题和注意力头数问题
def check_params(sampled_params):
    if sampled_params['hidden_size']%sampled_params['num_attention_heads'] !=0:
        return False
    # if not check_model_size(sampled_params):
    #     return False
    mapfunction = []
    for i in range(1, int(sampled_params['num_hidden_layers']) + 1):  # 筛选有效的mapfunction
        key = f'mapfunction_{i}'
        if key in sampled_params:
            mapfunction.append(int(sampled_params[key]))
    # 排除所有为0的数据
    mapfunction = [x for x in mapfunction if x != 0]
    # 如果全为0 则判断为失败
    if not mapfunction:  # 如果mapfunction为空
        return False

    # 检查 mapfunction 是否严格递增
    if all(mapfunction[i] < mapfunction[i + 1] for i in range(len(mapfunction) - 1)):
        return True
    else:
        return False

# 随机采样20个配置，进行蒸馏过程，输出过程数据
def sample_params(args,space,hid_distil):
    # 定义一个伪目标函数
    def objective(params):
        return 0

    hyperparameters=[]
    sample_num=20
    while len(hyperparameters) < sample_num:
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=1
        )
        # 输出抽取的组合
        sampled_params = space_eval(space, best)
        if check_params(sampled_params):
            print(sampled_params)
            hyperparameters.append(sampled_params)
    # 蒸馏过程
    accs, f1s, pres, recs = distill(args, hyperparameters,hid_distil, eval=False, surrogate=False)
    # accs = [round(random.uniform(0.5, 0.6), 2) for _ in range(sample_num)]
    # f1s = [round(random.uniform(0.5, 0.6), 2) for _ in range(sample_num)]
    # pres = [round(random.uniform(0.5, 0.6), 2) for _ in range(sample_num)]
    # recs = [round(random.uniform(0.5, 0.6), 2) for _ in range(sample_num)]

    with open("sample_params_data.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Tokenizer", "Vocab Size", "Num Hidden Layers", "Hidden Size", "Hidden Act", "Hidden Dropout Prob", "Intermediate Size", "Num Attention Heads", "Attention Probs Dropout Prob", "Max Sequence Length", "Position Embedding Type", "Pred Learning Rate", "Batch Size","Loss Function","Hid Learning Rate","Hid Epoches",'mapfunction_1','mapfunction_2','mapfunction_3','mapfunction_4','mapfunction_5','mapfunction_6','mapfunction_7','mapfunction_8','mapfunction_9','mapfunction_10','mapfunction_11','mapfunction_12',"Accuracy", "F1", "Precision", "Recall"])
        for d, acc,f1,pre,rec in zip(hyperparameters, accs, f1s, pres, recs):
            writer.writerow(Hyperparameters_convert(d) + [acc] + [f1] + [pre] + [rec])
    return hyperparameters,accs


# 目标参数
def objective(hyperparameters):
    '''Returns validation score from hyperparameters'''
    # 判断参数是否符合规则
    if not check_params(hyperparameters):
        return {'loss': 1, 'status': 'fail', 'params': hyperparameters}
    start_time = time.time()
    print(list(hyperparameters.values()))
    accs = surrogate_model_acc.predict([list(hyperparameters.values())])[0]
    object_loss = 1 - accs

    # 记录本次迭代的结果
    iteration = len(bayes_trials.trials)
    run_time = time.time() - start_time
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([hyperparameters, iteration, accs, object_loss, run_time])
    of_connection.close()

    return   {'loss': object_loss, 'status': 'ok', 'params': hyperparameters}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        required=False,
                        help="The student model dir.")
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        required=False,
                        help="The teacher model dir.")
    parser.add_argument("--tokenizer",
                        default=None,
                        type=str,
                        required=False,
                        help="The teacher model dir.")
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")

    parser.add_argument("--hid_epoches", type=int, default=42,
                        help="epoch num for hid training")
    parser.add_argument("--loss_function", type=str, default="mse",
                        help="loss function for attention")
    parser.add_argument("--pred_epoches", type=int, default=42,
                        help="epoch num for pred training")
    parser.add_argument("--hid_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for hid Adam.")
    parser.add_argument("--pred_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for pred Adam.")

    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--temperature',
                        type=float,
                        default=1.)
    parser.add_argument('--iteration',
                        type=int,
                        default=50)
    # prepare the device
    args = parser.parse_args()
    logger.info(args)
    args.device = torch.device("cuda:0")



    space = {
        # old config
        'tokenizer': hp.choice('tokenizer',[5]),#  fixed
        'vocab_size': hp.choice('vocab_size', [50265]),#  fixed
        # 'vocab_size': hp.quniform('vocab_size', 1000, 46000, 1000),
        'num_hidden_layers': hp.quniform('num_hidden_layers', 1, 12, 1),
        'hidden_size': hp.quniform('hidden_size', 16, 256, 16),
        'hidden_act': hp.choice('hidden_act',[1,2,3,4]),
        'hidden_dropout_prob': hp.choice('hidden_dropout_prob',[0.1, 0.2, 0.3, 0.4, 0.5]),
        'intermediate_size': hp.quniform('intermediate_size', 32, 3072, 32),
        'num_attention_heads': hp.choice('num_attention_heads',[12]),#  fixed
        'attention_probs_dropout_prob': hp.choice('attention_probs_dropout_prob',[0.1, 0.2, 0.3, 0.4, 0.5]),
        'max_sequence_length': hp.quniform('max_sequence_length', 256, 512, 1),
        'position_embedding_type': hp.choice('position_embedding_type',[1,2,3]),
        'pred_learning_rate': hp.choice('pred_learning_rate',[1e-3, 1e-4, 5e-5]),
        'batch_size': hp.choice('batch_size',[8,16]),
        # new config
        'loss_function': hp.choice('loss_function',[1, 2]),
        'hid_learning_rate': hp.choice('hid_learning_rate',[1e-3, 1e-4, 5e-5]),
        'hid_epoches': hp.quniform('hid_epoches', 4, 13, 1),
        'mapfunction_1': hp.choice('mapfunction_1', [0, hp.quniform('mapfunction_1_opt', 1, 12, 1)]),
        'mapfunction_2': hp.choice('mapfunction_2', [0, hp.quniform('mapfunction_2_opt', 1, 12, 1)]),
        'mapfunction_3': hp.choice('mapfunction_3', [0, hp.quniform('mapfunction_3_opt', 1, 12, 1)]),
        'mapfunction_4': hp.choice('mapfunction_4', [0, hp.quniform('mapfunction_4_opt', 1, 12, 1)]),
        'mapfunction_5': hp.choice('mapfunction_5', [0, hp.quniform('mapfunction_5_opt', 1, 12, 1)]),
        'mapfunction_6': hp.choice('mapfunction_6', [0, hp.quniform('mapfunction_6_opt', 1, 12, 1)]),
        'mapfunction_7': hp.choice('mapfunction_7', [0, hp.quniform('mapfunction_7_opt', 1, 12, 1)]),
        'mapfunction_8': hp.choice('mapfunction_8', [0, hp.quniform('mapfunction_8_opt', 1, 12, 1)]),
        'mapfunction_9': hp.choice('mapfunction_9', [0, hp.quniform('mapfunction_9_opt', 1, 12, 1)]),
        'mapfunction_10': hp.choice('mapfunction_10', [0, hp.quniform('mapfunction_10_opt', 1, 12, 1)]),
        'mapfunction_11': hp.choice('mapfunction_11', [0, hp.quniform('mapfunction_11_opt', 1, 12, 1)]),
        'mapfunction_12': hp.choice('mapfunction_12', [0, hp.quniform('mapfunction_12_opt', 1, 12, 1)]),
    }


    # old_space = {
    #     # old config 
    #     'tokenizer': hp.choice('tokenizer',[1,2,3,4]),#  "Byte-Pair Encoding", "WordPiece", "Unigram", "Word"
    #     # 'tokenizer': hp.choice('tokenizer',[5]),#fixed
    #     'vocab_size': hp.quniform('vocab_size', 1000, 46000, 1000),
    #     # 'vocab_size': hp.choice('vocab_size', [50265]),#  fixed
    #     'num_hidden_layers': hp.quniform('num_hidden_layers', 1, 12, 1),
    #     'hidden_size': hp.quniform('hidden_size', 16, 256, 16),
    #     'hidden_act': hp.choice('hidden_act',[1,2,3,4]), #"GELU", "ReLU", "SiLU", "GELU_new"
    #     'hidden_dropout_prob': hp.choice('hidden_dropout_prob',[0.1, 0.2, 0.3, 0.4, 0.5]),
    #     'intermediate_size': hp.quniform('intermediate_size', 32, 3072, 32),
    #     'num_attention_heads': hp.quniform('num_attention_heads', 1, 12, 1),
    #     # 'num_attention_heads': hp.choice('num_attention_heads',[12]),#  fixed
    #     'attention_probs_dropout_prob': hp.choice('attention_probs_dropout_prob',[0.1, 0.2, 0.3, 0.4, 0.5]),
    #     'max_sequence_length': hp.quniform('max_sequence_length', 256, 512, 1),
    #     'position_embedding_type': hp.choice('position_embedding_type',[1,2,3]),  #"absolute", "relative_key",  "relative_key_query"
    #     'pred_learning_rate': hp.choice('pred_learning_rate',[1e-3, 1e-4, 5e-5]),    
    #     'batch_size': hp.choice('batch_size',[8,16]),
    #     # new config 
    #     'loss_function': hp.choice('loss_function',[0]),
    #     'hid_learning_rate': hp.choice('hid_learning_rate',[0]),
    #     'hid_epoches': hp.choice('hid_epoches',[0]),
    #     'mapfunction_1': hp.choice('mapfunction_1', [1]),
    #     'mapfunction_2': hp.choice('mapfunction_2', [2]),
    #     'mapfunction_3': hp.choice('mapfunction_3', [3]),
    #     'mapfunction_4': hp.choice('mapfunction_4', [4]),
    #     'mapfunction_5': hp.choice('mapfunction_5', [5]),
    #     'mapfunction_6': hp.choice('mapfunction_6', [6]),
    #     'mapfunction_7': hp.choice('mapfunction_7', [7]),
    #     'mapfunction_8': hp.choice('mapfunction_8', [8]),
    #     'mapfunction_9': hp.choice('mapfunction_9', [9]),
    #     'mapfunction_10': hp.choice('mapfunction_10', [10]),
    #     'mapfunction_11': hp.choice('mapfunction_11', [11]),
    #     'mapfunction_12': hp.choice('mapfunction_12', [12]),
    # }

    params,accs=sample_params(args,space,True)
    # params, accs = sample_params(args, old_space,False)
    params= [list(d.values()) for d in params]
    
    # # if exist params and accs, then use them to train surrogate model
    # # 读取CSV文件
    # df = pd.read_csv('sample_params_data.csv')

    # # 将每行数据转换为列表，并存储在一个列表中
    # data_list = df.values.tolist()

    # params,accs = [data[:28] for data in data_list],[data[-4] for data in data_list]
    # for param,acc in zip(params, accs):
    #     print(param,acc)
    # params = [ParametersCSV_convert(param) for param in params]


    surrogate_model_acc = predictor([params, accs])
    # File to save first results
    out_file = 'gbm_trials.csv'
    of_connection = open(out_file, 'w')
    writer = csv.writer(of_connection)
    # Write the headers to the file
    writer.writerow(['hyperparameters', 'iteration', 'accs', 'object_loss', 'run_time'])
    of_connection.close()

    # Trials object to track progress
    bayes_trials = Trials()

    # Optimize
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=args.iteration, trials=bayes_trials)
    # best = fmin(fn=objective, space=old_space, algo=tpe.suggest, max_evals=args.iteration, trials=bayes_trials)
    
    best_params = bayes_trials.best_trial['result']['params']
    best_loss = bayes_trials.best_trial['result']['loss']
    # 将结果写入文件
    result_data = {
        'best_hyperparameters': best_params,
        'best_loss': best_loss
    }

    with open('bayes_optimization_results.json', 'w') as f:
        json.dump(result_data, f, indent=4)

    print("优化结果已记录在 bayes_optimization_results.json 文件中")