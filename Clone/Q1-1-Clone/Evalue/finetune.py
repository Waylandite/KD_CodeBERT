from ast import arg
import os
import torch
import logging
import argparse
import warnings
import numpy as np
from torch.nn import CrossEntropyLoss

from tqdm import tqdm
from model import Model
import torch.nn.functional as F
from utils import set_seed, load_and_cache_examples
from sklearn.metrics import recall_score, precision_score, f1_score
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaModel, RobertaTokenizer, \
    RobertaForSequenceClassification

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)



import time
def evaluate(args, model, tokenizer):
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, test=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size, num_workers=8, pin_memory=True)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    logits = []
    labels = []
    time_count = []
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for batch in bar:
        bar.set_description("evaluation")
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            time_start = time.time()
            logit = model(inputs,mod="eval")
            logit = F.softmax(logit)
            time_end = time.time()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
            time_count.append(time_end-time_start)
    print(sum(time_count)/len(time_count))

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)

    logits = F.softmax(torch.FloatTensor(logits))
    y_preds = logits[:, 1] > 0.5
    y_preds = y_preds.numpy()
    recall = recall_score(labels, y_preds)
    precision = precision_score(labels, y_preds)
    f1 = f1_score(labels, y_preds)
    result = {
        "eval_acc": np.mean(labels == y_preds),
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1)
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pred_model_dir", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default="../checkpoints", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--block_size", default=400, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--epochs", type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")


    args = parser.parse_args()
    args.device = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    logger.info("Device: %s, n_gpu: %s", args.device, args.n_gpu)
    set_seed(args.seed)

    args.model_name = "microsoft/codebert-base"

    tokenizer = RobertaTokenizer.from_pretrained('/data/wuruifeng/models/roberta-base')
    tokenizer.do_lower_case = True

    # config = RobertaConfig.from_pretrained(args.pred_model_dir)
    # model = Model(RobertaModel.from_pretrained(args.pred_model_dir,config=config))
    model=torch.load(args.pred_model_dir+"/model.pkl")
    logger.info("Training parameters %s", args)

    model.to(args.device)
    evaluate(args, model, tokenizer)




if __name__ == "__main__":
    main()
