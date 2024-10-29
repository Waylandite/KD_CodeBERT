import os
import json
import time
import torch
import logging
import argparse
import warnings
import numpy as np
from torch.nn import MSELoss
import torch.nn.functional as F
from tqdm import tqdm
from model import Model
from utils import set_seed, load_and_cache_examples
from sklearn.metrics import recall_score, precision_score, f1_score
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaForSequenceClassification, \
    RobertaTokenizer, RobertaModel

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def distill_loss(predicts, targets, temperature=1.0):
    predicts = predicts / temperature
    targets = targets / temperature
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return (- targets_prob * student_likelihood).mean() * temperature ** 2

def train(args, teacher_model,student_model, tokenizer):
    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader)/2
    teacher_model.to(args.device)
    student_model.to(args.device)
    #计算教师模型和学生模型的参数量
    size = 0
    for n, p in student_model.named_parameters():
        logger.info('n: {}'.format(n))
        size += p.nelement()
    logger.info('student Total parameters: {}'.format(size))
    size = 0
    for n, p in teacher_model.named_parameters():
        logger.info('n: {}'.format(n))
        size += p.nelement()
    logger.info('teacher Total parameters: {}'.format(size))


    # 准备优化器
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in student_model.named_parameters(
        ) if not any(nd in n for nd in no_decay)]}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)

    # 开始训练
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d",args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    best_acc = 0
    best_f1 = 0
    loss_mse = MSELoss()

    student_model.zero_grad()
    for idx in range(0, int(args.epochs)):
        student_model.train()
        tr_num = 0
        train_loss = 0
        tr_att_loss = 0.
        tr_rep_loss = 0.
        tr_cls_loss = 0.
        logger.info("Epoch [{}/{}]".format(idx + 1, args.epochs))
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        bar.set_description("Train")
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)
            student_outputs = student_model(inputs, is_student=True)
            student_logits = student_outputs.get('logits')
            student_reps = student_outputs.get('hidden_states')
            student_atts = student_outputs.get('attentions')

            att_loss = 0.
            rep_loss = 0.
            cls_loss = 0.
            with torch.no_grad():
                teacher_outputs =teacher_model(inputs)
                teacher_logits=teacher_outputs.get('logits')
                teacher_reps=teacher_outputs.get('hidden_states')
                teacher_atts=teacher_outputs.get('attentions')

            # 隐藏层蒸馏
            if  args.hid_distill:
                teacher_layer_num = len(teacher_atts)
                student_layer_num = len(student_atts)
                assert teacher_layer_num % student_layer_num == 0
                layers_per_block = int(teacher_layer_num / student_layer_num)
                # new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]for i in range(student_layer_num)]
                new_teacher_atts = [teacher_atts[i+6]for i in range(student_layer_num)]
                # new_teacher_atts = [teacher_atts[i]for i in range(student_layer_num)]

                for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(args.device),
                                              student_att)

                    teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(args.device),
                                              teacher_att)

                    tmp_loss = loss_mse(student_att, teacher_att)

                    att_loss += tmp_loss

                # new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
                
                new_teacher_reps = [teacher_reps[i+6] for i in range(student_layer_num + 1)]
                new_teacher_reps[0] = teacher_reps[0]
                # new_teacher_reps = [teacher_reps[i] for i in range(student_layer_num + 1)]
                new_student_reps = student_reps
                for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                    tmp_loss = loss_mse(student_rep, teacher_rep)
                    rep_loss += tmp_loss

                loss = rep_loss + att_loss
                tr_att_loss += att_loss.item()
                tr_rep_loss += rep_loss.item()
            # 输出层蒸馏/传统蒸馏
            else:
                cls_loss = distill_loss(student_logits, teacher_logits, temperature=args.temperature)
                loss = cls_loss
                tr_cls_loss += cls_loss.item()



            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)


            tr_num += 1
            train_loss += loss.item()
            avg_loss = round(train_loss/tr_num, 5)
            avg_cls_loss=round(tr_cls_loss/tr_num, 5)
            avg_att_loss=round(tr_att_loss/tr_num, 5)
            avg_rep_loss=round(tr_rep_loss/tr_num, 5)

            bar.set_description("epoch {} loss {} cls_loss {} att_loss {} rep_loss {}".format(idx, avg_loss, avg_cls_loss, avg_att_loss, avg_rep_loss))
            # bar.set_description("epoch {} loss {}".format(idx, avg_loss))
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    if args.pred_distill or args.tradition_distill:
                        results = evaluate(args, student_model, tokenizer)
                        logger.info("  "+"*"*20)
                        logger.info("  Current ACC:%s", round(results["eval_acc"], 4))
                        logger.info("  Best ACC:%s", round(best_acc, 4))
                        logger.info("  Best F1:%s", round(best_f1, 4))
                        logger.info("  "+"*"*20)


                    if args.hid_distill:
                        save_model = True
                    else:
                        save_model = False
                        # if results["eval_f1"] >= best_f1:
                        if (results["eval_acc"] > best_acc):
                            best_f1 = results["eval_f1"]
                            best_acc = results["eval_acc"]
                            save_model = True

                    if save_model:
                        if args.pred_distill:
                            checkpoint_prefix = "pred_distill"
                        elif args.hid_distill:
                            checkpoint_prefix = "hidden_distill"
                        else:
                            checkpoint_prefix = "tradition_distill"

                        output_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        output_config_file = os.path.join(output_dir, "config.json")

                        model_to_save = student_model.encoder
                        # torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        torch.save(student_model, output_dir + "/model.pkl")
                        logger.info("Saving model checkpoint to %s", output_dir)
                    else:
                        logger.info("Model checkpoint are not saved")


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
            logit = model(inputs, mod="eval")
            logit = F.softmax(logit)
            time_end = time.time()
            time_count.append(time_end-time_start)
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
            time_count.append(time_end - time_start)
    print(sum(time_count) / len(time_count))
    logger.info("  Time used = %f s", sum(time_count)/len(time_count))
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
    parser.add_argument('--hid_distill',
                        action='store_true')
    parser.add_argument('--pred_distill',
                        action='store_true')
    parser.add_argument('--tradition_distill',
                        action='store_true')
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--output_dir", default="../checkpoints", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--block_size", default=400, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument("--epochs", type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--temperature',
                        type=float,
                        default=1.)


    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    args = parser.parse_args()
    logger.info(args)
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.output_mode = "classification"
    logger.info("Device: %s, n_gpu: %s", args.device, args.n_gpu)
    set_seed(args.seed)


    #准备tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('/data/wuruifeng/models/roberta-base')
    tokenizer.do_lower_case = True

    # 准备教师模型
    # teacher_config = RobertaConfig.from_pretrained(args.teacher_model)
    # teacher_config.num_labels = 2
    teacher_model = torch.load(args.teacher_model+"/model.pkl")

    # 准备学生模型
    if args.hid_distill or args.tradition_distill:
        student_config = RobertaConfig.from_pretrained(args.student_model)
        student_config.num_labels = 2
        student_model = Model(RobertaModel(student_config))
    else:
        # student_config = RobertaConfig.from_pretrained(args.student_model)
        # student_config.num_labels = 2
        # student_model = Model(RobertaModel.from_pretrained(args.student_model,config=student_config))
        student_model =torch.load(args.student_model+"/model.pkl")

    logger.info("Training/evaluation parameters %s", args)


    # train(args, teacher_model,student_model,tokenizer)


if __name__ == "__main__":
    main()