import os
import json
import time
import torch
import logging
import argparse
import warnings
import numpy as np

from tqdm import tqdm
from model import Model
from utils import set_seed, TextDataset
from sklearn.metrics import recall_score, precision_score, f1_score
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
import torch.nn.functional as F
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)



def evaluate(args, model, tokenizer, eval_when_training=False):
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size, num_workers=8, pin_memory=True)

    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

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
            logit = model(inputs)
            logit = F.softmax(logit)
            time_end = time.time()
            time_count.append(time_end-time_start)
            logits.append(logit.cpu().numpy())
        labels.append(label.cpu().numpy())
    logger.info("  Time used = %f s", sum(time_count)/len(time_count))
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)

    if args.do_inference:
        folder = "/".join(args.eval_data_file.split("/")[:-1])
        data = []
        with open(args.eval_data_file) as f:
            for line in f:
                data.append(json.loads(line.strip()))

        new_data = []
        for d, p in zip(data, logits.tolist()):
            d["soft_label"] = p
            new_data.append(d)

        with open(os.path.join(folder, "soft_label_train_codebert.jsonl"), "w") as f:
            for d in new_data:
                f.write(json.dumps(d) + "\n")

        logger.info("Saving inference results to %s", os.path.join(folder, "soft_label_train_codebert.jsonl"))
        return None

    preds = logits[:, 0] > 0.5
    eval_acc = np.mean(labels==preds)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels, preds)

    result = {
        "eval_acc": eval_acc,
        "eval_precision": float(precision),
        "eval_recall": float(recall),
        "eval_f1": float(f1)
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The  model dir.")
    parser.add_argument("--output_dir", default="../checkpoints", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,required=True,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_inference", action="store_true",
                        help="Whether to run inference for unlabeled data.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Run evaluation during training at each logging step.")

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
    parser.add_argument("--epoch", type=int, default=42,
                        help="random seed for initialization")

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    args = parser.parse_args()
    logger.info(args)
    args.device = torch.device("cuda")
    # args.device = torch.device(
    #     "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.per_gpu_train_batch_size = args.train_batch_size//args.n_gpu
    args.per_gpu_eval_batch_size = args.eval_batch_size//args.n_gpu

    logger.info("Device: %s, n_gpu: %s", args.device, args.n_gpu)

    set_seed(args.seed)

    tokenizer = RobertaTokenizer.from_pretrained('/data/wuruifeng/models/roberta-base')
    tokenizer.do_lower_case = True

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)


    config = RobertaConfig.from_pretrained(args.pred_model_dir)
    config.num_labels = 2
    model = Model(RobertaForSequenceClassification.from_pretrained(args.pred_model_dir, config=config))

    # model.load_state_dict(torch.load(args.pred_model_dir+"/pytorch_model.bin"), strict=False)
    model.to(args.device)
    params = sum(p.numel() for p in model.parameters())
    logger.info("size %f", params)

    evaluate(args, model, tokenizer)


if __name__ == "__main__":
    main()
