import os
import time
import torch
import logging
import warnings
import numpy as np
import torch.nn.functional as F
from torch.nn import MSELoss
from tqdm import tqdm
from utils import Hyperparameters_convert, distill_loss, TextDataset, set_seed
from models import Model
from sklearn.metrics import recall_score, precision_score, f1_score
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaModel, \
    RobertaForSequenceClassification, RobertaTokenizer
import argparse
warnings.filterwarnings("ignore")
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(student_model, teacher_model, map_function, train_dataloader, eval_dataloader, hid_epoches, pred_epoches,
          hid_learning_rate, pred_learning_rate, temperature, device, loss_function, surrogate=False):
    # hid_distil

    total_params = sum(p.numel() for p in student_model.parameters())
    logger.info(f"{total_params:,} total parameters.")
    logger.info(f"{total_params * 4 / 1e6} MB model size")
    num_steps = len(train_dataloader) * hid_epoches
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)]}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=hid_learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_steps * 0.1,
                                                num_training_steps=num_steps)
    logger.info("***** Running hid  distil training *****")

    # Prepare loss functions
    loss_mse = MSELoss()

    def attention_kl_divergence(student_scores, teacher_scores):
        # Compute the softmax probabilities along the last dimension of the tensors
        teacher_probs = torch.nn.functional.softmax(teacher_scores, dim=-1)
        student_log_probs = torch.nn.functional.log_softmax(student_scores, dim=-1)

        # Compute the KL divergence between the two distributions
        kl_divergence = torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

        return kl_divergence

    for epoch in range(hid_epoches):
        student_model.train()
        tr_num = 0
        train_loss = 0
        tr_att_loss = 0.
        tr_rep_loss = 0.

        logger.info("Epoch [{}/{}]".format(epoch + 1, hid_epoches))
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        # bar.set_description("Train")
        for batch in bar:
            att_loss = 0.
            rep_loss = 0.
            inputs = batch[0].to(device)
            student_outputs = student_model(inputs, is_student=True)
            student_reps = student_outputs.get('hidden_states')
            student_atts = student_outputs.get('attentions')

            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
                teacher_reps = teacher_outputs.get('hidden_states')
                teacher_atts = teacher_outputs.get('attentions')
            # calcalate new teacher hidden state

            # important!  atts calculate

            new_teacher_atts = []
            new_student_atts = []
            for index, x in enumerate(map_function):
                if x != 0:
                    new_student_atts.append(student_atts[index])
                    new_teacher_atts.append(teacher_atts[x - 1])

            for student_att, teacher_att in zip(new_student_atts, new_teacher_atts):
                student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                          student_att)

                teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                          teacher_att)

                if loss_function == 'kl':  
                    tmp_loss = attention_kl_divergence(student_att, teacher_att)
                elif loss_function == 'mse':
                    tmp_loss = loss_mse(student_att, teacher_att)
                else:
                    print("loss function error")
                    return
                att_loss += tmp_loss

            # important! reps calculate
            new_teacher_reps = []
            new_student_reps = []
            new_teacher_reps.append(teacher_reps[0])
            new_student_reps.append(student_reps[0])
            for index, x in enumerate(map_function):
                if x != 0:
                    new_student_reps.append(student_reps[index + 1])
                    new_teacher_reps.append(teacher_reps[x])

            for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                tmp_loss = loss_mse(student_rep, teacher_rep)
                rep_loss += tmp_loss

            loss = rep_loss + att_loss
            loss.backward()

            train_loss += loss.item()
            tr_att_loss += att_loss.item()
            tr_rep_loss += rep_loss.item()

            tr_num += 1
            avg_loss = round(train_loss / tr_num, 5)
            avg_cls_loss = 0
            avg_att_loss = round(tr_att_loss / tr_num, 5)
            avg_rep_loss = round(tr_rep_loss / tr_num, 5)
            bar.set_description(
                "epoch {} loss {} cls_loss {} att_loss {} rep_loss {}".format(epoch, avg_loss, avg_cls_loss,
                                                                              avg_att_loss, avg_rep_loss))

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        logger.info("Train Loss: {0}, Atts Loss: {1}, Rep Loss: {2}".format(train_loss / tr_num, tr_att_loss / tr_num,
                                                                            tr_rep_loss / tr_num))
    num_steps = len(train_dataloader) * pred_epoches
    no_decay = ["bias", "LayerNorm.weight"]
    total_params = sum(p.numel() for p in student_model.parameters())
    logger.info(f"{total_params:,} total parameters.")
    logger.info(f"{total_params * 4 / 1e6} MB model size")
    optimizer_grouped_parameters = [
        {"params": [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)]}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=pred_learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_steps * 0.1,
                                                num_training_steps=num_steps)
    dev_best_acc = 0
    dev_best_f1 = 0
    dev_best_precision = 0
    dev_best_recall = 0

    logger.info("***** Running pred distil training *****")

    for epoch in range(pred_epoches):
        student_model.train()
        tr_num = 0
        train_loss = 0
        tr_cls_loss = 0

        logger.info("Epoch [{}/{}]".format(epoch + 1, pred_epoches))
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        # bar.set_description("Train")
        for batch in bar:
            inputs = batch[0].to(device)
            student_outputs = student_model(inputs, is_student=True)
            student_logits = student_outputs.get('logits')

            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
                teacher_logits = teacher_outputs.get('logits')

            cls_loss = distill_loss(student_logits, teacher_logits, temperature=temperature)

            loss = cls_loss
            loss.backward()

            train_loss += loss.item()
            tr_cls_loss = cls_loss.item()
            tr_num += 1

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            avg_loss = round(train_loss / tr_num, 5)
            avg_cls_loss = round(tr_cls_loss / tr_num, 5)
            avg_att_loss = 0
            avg_rep_loss = 0
            bar.set_description(
                "epoch {} loss {} cls_loss {} att_loss {} rep_loss {}".format(epoch, avg_loss, avg_cls_loss,
                                                                              avg_att_loss, avg_rep_loss))

        dev_results = evaluate(student_model, device, eval_dataloader)
        dev_acc = dev_results["eval_acc"]
        dev_f1 = dev_results["eval_f1"]
        dev_precision = dev_results["eval_precision"]
        dev_recall = dev_results["eval_recall"]
        if dev_f1 >= dev_best_f1:
            dev_best_f1 = dev_f1
        if dev_precision >= dev_best_precision:
            dev_best_precision = dev_precision
        if dev_recall >= dev_best_recall:
            dev_best_recall = dev_recall
        if dev_acc >= dev_best_acc:
            dev_best_acc = dev_acc
            if surrogate:
                output_dir = os.path.join("./checkpoints", "BestAcc")
                os.makedirs(output_dir, exist_ok=True)
                torch.save(student_model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                logger.info("New best model found and saved.")

        logger.info("Train Loss: {0}, Val Acc: {1}, Val Precision: {2}, Val Recall: {3}, Val F1: {4}".format(
            train_loss / tr_num, dev_results["eval_acc"], dev_results["eval_precision"], dev_results["eval_recall"],
            dev_results["eval_f1"]))

    logger.info("***** one calcalatitae is done! *****")
    logger.info("Best Acc: {0}, Best Precision: {1}, Best Recall: {2}, Best F1: {3}".format(dev_best_acc,
                                                                                            dev_best_precision,
                                                                                            dev_best_recall,
                                                                                            dev_best_f1))
    return {"eval_acc": dev_best_acc, "eval_f1": dev_best_f1, "eval_precision": dev_best_precision,
            "eval_recall": dev_best_recall}


def evaluate(model, device, eval_dataloader):
    model.eval()
    predict_all = []
    labels_all = []
    time_count = []
    with torch.no_grad():
        bar = tqdm(eval_dataloader, total=len(eval_dataloader))
        bar.set_description("Evaluation")
        for batch in bar:
            inputs = batch[0].to(device)
            label = batch[1].to(device)
            time_start = time.time()
            logit = model(inputs, mod="eval")
            time_end = time.time()

            logit = F.softmax(logit)
            predict_all.append(logit.cpu().numpy())
            labels_all.append(label.cpu().numpy())
            time_count.append(time_end - time_start)

    latency = np.mean(time_count)
    logger.info("Average Inference Time pre Batch: {}".format(latency))
    predict_all = np.concatenate(predict_all, 0)
    labels_all = np.concatenate(labels_all, 0)
    # vul
    preds = predict_all[:, 0] > 0.5
    #   clone
    # preds = predict_all[:, 1] > 0.5
    eval_acc = np.mean(labels_all == preds)
    recall = recall_score(labels_all, preds)
    precision = precision_score(labels_all, preds)
    f1 = f1_score(labels_all, preds)
    results = {
        "eval_acc": eval_acc,
        "eval_precision": float(precision),
        "eval_recall": float(recall),
        "eval_f1": float(f1),
        "inference_time": latency
    }
    logger.info("***** Eval results *****")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(round(results[key], 4)))
    return results


"""
    tokenizer: tokenizer
    args: arguments 
    map_functionList: list of map functions
    eval: whether to evaluate the model
    surrogate: whether to save the model
"""


def distill(args, hyperparametersList, eval=False, surrogate=False):

    set_seed(args.seed)
    # teacher model
    teacher_config = RobertaConfig.from_pretrained(args.teacher_model)
    teacher_config.num_labels = 2
    teacher_model = Model(RobertaForSequenceClassification.from_pretrained(args.teacher_model, config=teacher_config))
    teacher_model.to(args.device)
    # student model 
    student_config = RobertaConfig.from_pretrained(args.student_model)
    student_config.num_labels = 2

    dev_best_accs = []
    dev_best_f1s = []
    dev_best_pres = []
    dev_best_recs = []

    for hyperparam in hyperparametersList:
        
        tokenizer_type, vocab_size, num_hidden_layers, hidden_size, hidden_act, hidden_dropout_prob, intermediate_size, num_attention_heads, attention_probs_dropout_prob, max_sequence_length, position_embedding_type, pred_learning_rate, batch_size,loss_function,hid_learning_rate,hid_epoches,*rest = Hyperparameters_convert(hyperparam)
            
        student_config.vocab_size = vocab_size
        student_config.num_hidden_layers = num_hidden_layers
        student_config.hidden_size = hidden_size
        student_config.hidden_act = hidden_act
        student_config.hidden_dropout_prob = hidden_dropout_prob
        student_config.intermediate_size = intermediate_size
        student_config.num_attention_heads = num_attention_heads
        student_config.attention_probs_dropout_prob = attention_probs_dropout_prob
        student_config.max_position_embeddings = max_sequence_length+2
        student_config.position_embedding_type = position_embedding_type

        
        # prepare the tokenizer
        tokenizer = RobertaTokenizer.from_pretrained("/home/wuruifeng/data/models/roberta-base")
        tokenizer.do_lower_case = True

        # 准备训练数据
        train_dataset = TextDataset(tokenizer, max_sequence_length, file_path=args.train_data_file)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

        eval_dataset = TextDataset(tokenizer, max_sequence_length, file_path=args.eval_data_file)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=8,
                                        pin_memory=True)
        
        
        # 根据学生模型的层数,选择hyperparam中有效的map_function
        mapfunction = []
        for i in range(1, num_hidden_layers+1):  # 筛选有效的mapfunction
            key = f'mapfunction_{i}'
            if key in hyperparam:
                mapfunction.append(int(hyperparam[key]))

        student_model = Model(RobertaForSequenceClassification(student_config))
        student_model.to(args.device)
        print(hyperparam)
        dev_best_outcomes = train(student_model, teacher_model, mapfunction, train_dataloader, eval_dataloader,
                                  hid_epoches,
                                  args.pred_epoches,
                                  hid_learning_rate,
                                  pred_learning_rate,
                                  args.temperature,
                                  args.device,
                                  loss_function,
                                  surrogate=False)

        dev_best_accs.append(dev_best_outcomes["eval_acc"])
        dev_best_f1s.append(dev_best_outcomes["eval_f1"])
        dev_best_pres.append(dev_best_outcomes["eval_precision"])
        dev_best_recs.append(dev_best_outcomes["eval_recall"])

    return dev_best_accs, dev_best_f1s, dev_best_pres, dev_best_recs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model",
                        default="/home/wuruifeng/data/models/6layer_config",
                        type=str,
                        required=False,
                        help="The student model dir.")
    parser.add_argument("--teacher_model",
                        default="/home/wuruifeng/data/result/Vulnerability-Detection/Finetune",
                        type=str,
                        required=False,
                        help="The teacher model dir.")
    parser.add_argument("--train_data_file", default="/home/wuruifeng/data/data/Vulnerability-Detection/data/label_train.jsonl", type=str, required=False,
                        help="The input training data file (a text file).")
    parser.add_argument("--eval_data_file", default="/home/wuruifeng/data/data/Vulnerability-Detection/data/valid.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--pred_epoches", type=int, default=4,
                        help="epoch num for pred training")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--temperature',
                        type=float,
                        default=1.)
    parser.add_argument('--iteration',
                        type=int,
                        default=500)
    # prepare the device
    args = parser.parse_args()
    logger.info(args)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # CUDA_VISIBLE_DEVICES 表当前可被python程序检测到的显卡

    args.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")  # 多GPU时可指定起始位置/编号

    args.device=torch.device("cpu")

    hyper_args=[
    {
    'attention_probs_dropout_prob': 0.1, 
    'batch_size': 32,
    'hidden_act': 1, 
    'hidden_dropout_prob': 0.4,
    'hidden_size': 70.0, 
    'intermediate_size': 1710.0,
    'max_sequence_length': 512.0, 
    'num_attention_heads': 2.0, 
    'num_hidden_layers': 4.0, 
    'position_embedding_type': 3, 
    'pred_learning_rate': 0.001, 
    'tokenizer': 3, 
    'vocab_size': 27000.0,
    
    'loss_function': 2, 
    'hid_epoches': 12.0, 
    'hid_learning_rate': 0.0001,
    'mapfunction_1': 11.0, 'mapfunction_2': 0, 'mapfunction_3': 0, 'mapfunction_4': 0, 'mapfunction_5': 7.0, 'mapfunction_6': 0, 'mapfunction_7': 8.0, 'mapfunction_8': 0, 'mapfunction_9': 11.0, 'mapfunction_10': 10.0, 'mapfunction_11': 10.0, 'mapfunction_12': 0}

    ]

    accs, f1s, pres, recs = distill(args, hyper_args, eval=False, surrogate=False)
    print(accs)
