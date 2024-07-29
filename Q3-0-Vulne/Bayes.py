import argparse
import json
import random
import time

import numpy as np
import torch
from hyperopt import hp, space_eval
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
import csv
import logging
from utils import Hyperparameters_convert
from transformers import RobertaTokenizer
from surrogate import predictor

from distill_utils import distill

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
# 判断是否存在学习冲突问题
def check_params(sampled_params):
    mapfunction = []
    for i in range(1, int(sampled_params['hidden_layers']) + 1):  # 筛选有效的mapfunction
        key = f'mapfunction_{i}'
        if key in sampled_params:
            mapfunction.append(int(sampled_params[key]))
    # 排除所有为0的数据
    mapfunction = [x for x in mapfunction if x != 0]

    # 检查 mapfunction 是否严格递增
    if all(mapfunction[i] < mapfunction[i + 1] for i in range(len(mapfunction) - 1)):
        return True
    else:
        return False

# 随机采样20个配置，进行蒸馏过程，输出过程数据
def sample_params(args,space):
    # 定义一个伪目标函数
    def objective(params):
        return 0

    hyperparameters=[]
    while len(hyperparameters) < 20:
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
    # accs, f1s, pres, recs = distill(args, hyperparameters, eval=False, surrogate=False)
    accs = [round(random.uniform(0.5, 0.6), 2) for _ in range(20)]
    f1s = [round(random.uniform(0.5, 0.6), 2) for _ in range(20)]
    pres = [round(random.uniform(0.5, 0.6), 2) for _ in range(20)]
    recs = [round(random.uniform(0.5, 0.6), 2) for _ in range(20)]

    with open("sample_params_data.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["learning_rate","hid_epoch","hidden_layers",'mapfunction_1','mapfunction_2','mapfunction_3','mapfunction_4','mapfunction_5','mapfunction_6','mapfunction_7','mapfunction_8','mapfunction_9''mapfunction_10','mapfunction_11','mapfunction_12',"Accuracy", "F1", "Precision", "Recall"])
        for d, acc,f1,pre,rec in zip(hyperparameters, accs, f1s, pres, recs):
            writer.writerow(Hyperparameters_convert(d) + [acc] + [f1] + [pre] + [rec])
    return hyperparameters,accs


# 目标参数
def objective(hyperparameters):
    '''Returns validation score from hyperparameters'''
    # 判断参数是否符合规则
    # 判断参数是否符合规则
    if not check_params(hyperparameters):
        return {'loss': 1, 'status': 'fail', 'params': hyperparameters}
    start_time = time.time()

    accs = surrogate_model_acc.predict([Hyperparameters_convert(hyperparameters)])[0]
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
    args.device = torch.device("cuda")



    # Define the search space
    # parm1 learning_rate 学习率
    # parm2 hid_epoch 隐藏层蒸馏迭代次数
    # parm3 hidden_layers 隐藏层堆叠数
    # parm-else mapfunction_1-12 映射函数  0 means no learning
    space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.00005), np.log(0.001)),
        'hid_epoch': hp.quniform('hid_epoch', 8, 20, 1),
        'hidden_layers': hp.quniform('hidden_layers', 1, 12, 1),
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

    params,accs=sample_params(args,space)
    params=[Hyperparameters_convert(p) for p in params]
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
    best = fmin(fn=objective, space=space, algo=tpe.suggest,
                max_evals=args.iteration, trials=bayes_trials)
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