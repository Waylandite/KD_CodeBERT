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


def train(args, model, tokenizer):

    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    args.max_steps = args.epochs*len(train_dataloader)
    args.save_steps = len(train_dataloader)
    model.to(args.device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters(
        ) if not any(nd in n for nd in no_decay)]}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)



    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    best_acc = 0
    best_f1 = 0
    model.zero_grad()

    for idx in range(0, int(args.epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            model.train()
            output = model(inputs,mod="train")
            logits=output.get('logits')
            loss = CrossEntropyLoss()(logits, labels)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_num += 1
            train_loss += loss.item()
            avg_loss = round(train_loss/tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    results = evaluate(args, model, tokenizer)

                    logger.info("  "+"*"*20)
                    logger.info("  Current Acc:%s", round(results["eval_acc"], 4))
                    logger.info("  Best Acc:%s", round(best_acc, 4))
                    logger.info("  Best F1:%s", round(best_f1, 4))
                    logger.info("  "+"*"*20)

                    if results["eval_acc"] >= best_acc:
                        best_acc = results["eval_acc"]
                        best_f1 = results["eval_f1"]
                        output_dir =args.output_dir
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        output_config_file = os.path.join(output_dir, "config.json")
                        model_to_save = model.encoder
                        # torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        torch.save(model, args.output_dir+"/model.pkl")
                        logger.info("Saving model checkpoint to %s", output_dir)
                    else:
                        logger.info("Model checkpoint are not saved")

import time
def evaluate(args, model, tokenizer):
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
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

    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default="../checkpoints", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
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
    parser.add_argument("--learning_rate", default=2e-5, type=float,
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
    # config = RobertaConfig.from_pretrained('/data/wuruifeng/models/codebert-base')
    config = RobertaConfig.from_pretrained('/data/wuruifeng/models/4layer_config')
    tokenizer = RobertaTokenizer.from_pretrained('/data/wuruifeng/models/roberta-base')
    tokenizer.do_lower_case = True


    # model = Model(RobertaModel.from_pretrained('/data/wuruifeng/models/codebert-base',config=config))
    model = Model(RobertaModel(config))
    logger.info("Training parameters %s", args)

    model.to(args.device)
    train(args, model, tokenizer)




if __name__ == "__main__":
    main()
