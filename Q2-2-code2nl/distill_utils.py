from itertools import cycle
import os
import time
import torch
import logging
import warnings
import random
import bleu
import numpy as np
import torch.nn.functional as F
from torch.nn import MSELoss
import torch.nn as nn
from tqdm import tqdm
from utils import convert_examples_to_features, distill_loss, TextDataset, load_and_cache_examples, read_examples, set_seed
from model import Model, StudentSeq2Seq, TeacherSeq2Seq
from sklearn.metrics import recall_score, precision_score, f1_score
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler,TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaModel, \
    RobertaForSequenceClassification



warnings.filterwarnings("ignore")
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(student_model, teacher_model, map_function, train_dataloader, eval_dataloader,bleu_eval_dataloader,bleu_eval_examples, hid_epoches, pred_epoches,
          hid_learning_rate, pred_learning_rate, temperature, device, loss_function,surrogate=False):

    total_params = sum(p.numel() for p in student_model.parameters())
    logger.info(f"{total_params:,} total parameters.")
    logger.info(f"{total_params * 4 / 1e6} MB model size")
    
    # hid_distil
    num_train_optimization_steps =  hid_epoches

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)]}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=hid_learning_rate,eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0 * 0.1,
                                                num_training_steps=num_train_optimization_steps)
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

    tr_loss, tr_att_loss, tr_rep_loss , nb_tr_steps = 0, 0, 0, 0, 0

    bar = tqdm(range(num_train_optimization_steps), total=num_train_optimization_steps)
    train_dataloader = cycle(train_dataloader)
    eval_flag = True
    
    for step in bar:
        att_loss, rep_loss = 0, 0
        student_model.train()
        batch = next(train_dataloader)
        batch = tuple(t.to(device) for t in batch)
        source_ids, source_mask, target_ids, target_mask = batch
        student_logits, student_reps, student_atts,student_encoder_output = student_model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids,
                            target_mask=target_mask,is_student=True)
        with torch.no_grad():
            teacher_logits, teacher_reps, teacher_atts,teacher_encoder_output = teacher_model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids,
                            target_mask=target_mask)
        # pred distil detail 
        new_teacher_atts = []
        new_student_atts = []
        for index,x in enumerate(map_function):
            if x !=0:
                new_student_atts.append(student_atts[index])
                new_teacher_atts.append(teacher_atts[x-1])
        
        for student_att, teacher_att in zip(new_student_atts, new_teacher_atts):
            student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                    student_att)

            teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                    teacher_att)

            if loss_function == "kl":
                tmp_loss = attention_kl_divergence(student_att, teacher_att)
            else:
                tmp_loss = loss_mse(student_att, teacher_att)

            att_loss += tmp_loss

        # important! reps calculate
        new_teacher_reps = []
        new_student_reps = []
        new_teacher_reps.append(teacher_reps[0])
        new_student_reps.append(student_reps[0])
        for index,x in enumerate(map_function):
            if x !=0:
                new_student_reps.append(student_reps[index+1])
                new_teacher_reps.append(teacher_reps[x])

        for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
            tmp_loss = loss_mse(student_rep, teacher_rep)
            rep_loss += tmp_loss

        loss = rep_loss + att_loss
        tr_loss += loss.item()
        tr_att_loss += att_loss.item()
        tr_rep_loss += rep_loss.item()
        
        avg_train_loss = round(tr_loss / (nb_tr_steps + 1), 4)
        avg_cls_loss=0
        avg_att_loss=round(tr_att_loss/(nb_tr_steps + 1), 5)
        avg_rep_loss=round(tr_rep_loss/(nb_tr_steps + 1), 5)
        bar.set_description(" loss {} cls_loss {} att_loss {} rep_loss {}".format(avg_train_loss, avg_cls_loss, avg_att_loss, avg_rep_loss))

        nb_tr_steps += 1
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()




    logger.info("***** Running pred distil training *****")

    no_decay = ["bias", "LayerNorm.weight"]
    total_params = sum(p.numel() for p in student_model.parameters())
    logger.info(f"{total_params:,} total parameters.")
    logger.info(f"{total_params * 4 / 1e6} MB model size")
    optimizer_grouped_parameters = [
        {"params": [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)]}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=pred_learning_rate,eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0 * 0.1,
                                                num_training_steps=num_train_optimization_steps)


    nb_tr_steps, tr_loss, best_bleu, best_ppl =  0, 0, 0,  1e6
    bar = tqdm(range(num_train_optimization_steps), total=num_train_optimization_steps)
    train_dataloader = cycle(train_dataloader)
    eval_flag = True
    for step in bar:
        cls_loss=0.
        student_model.train()
        batch = next(train_dataloader)
        batch = tuple(t.to(device) for t in batch)
        source_ids, source_mask, target_ids, target_mask = batch
        student_logits, student_reps, student_atts,student_encoder_output = student_model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids,
                            target_mask=target_mask,is_student=True)
        with torch.no_grad():
            teacher_logits, teacher_reps, teacher_atts,teacher_encoder_output = teacher_model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids,
                            target_mask=target_mask)
        # cls_loss=distill_loss(student_logits,teacher_logits)

        # Shift so that tokens < n predict n
        active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        # soft label
        encoder_loss=distill_loss(shift_logits,shift_teacher_logits)
        # hard label
        decoder_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                    shift_labels.view(-1)[active_loss])
        cls_loss=encoder_loss+decoder_loss

        loss = cls_loss
        
        tr_loss += loss.item()
        avg_train_loss = round(tr_loss / (nb_tr_steps + 1), 4)
        avg_cls_loss=round(tr_loss /(nb_tr_steps + 1), 5)
        avg_att_loss=0
        avg_rep_loss=0

        bar.set_description(" loss {} cls_loss {} att_loss {} rep_loss {}".format(avg_train_loss, avg_cls_loss, avg_att_loss, avg_rep_loss))

        nb_tr_steps += 1
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if (nb_tr_steps, + 1) % args.eval_steps == 0:
            dev_results = evaluate(student_model, device, eval_dataloader,bleu_eval_dataloader, bleu_eval_examples)



    logger.info("***** one calcalatitae is done! *****")
    logger.info("best_bleu: {0}, best_ppl: {1}".format(best_bleu, best_ppl))
    return {"best_bleu": best_bleu, "best_ppl": best_ppl}


def evaluate(model, device, eval_dataloader,bleu_eval_dataloader, bleu_eval_examples):
    model.eval()
    eval_loss, tokens_num = 0, 0
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        source_ids, source_mask, target_ids, target_mask = batch

        with torch.no_grad():
            _, loss, num = model(source_ids=source_ids, source_mask=source_mask,
                                    target_ids=target_ids, target_mask=target_mask)
        eval_loss += loss.sum().item()
        tokens_num += num.sum().item()
    # Pring loss of dev dataset
    model.train()
    eval_loss = eval_loss / tokens_num
    eval_ppl=round(np.exp(eval_loss), 5)
    logger.info("  %s = %s " % ("eval_ppl", str(eval_ppl)))
    logger.info("  " + "*" * 20)
    # Calculate bleu
    model.eval()
    p = []
    for batch in bleu_eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        source_ids, source_mask = batch
        with torch.no_grad():
            preds = model(source_ids=source_ids, source_mask=source_mask)
            for pred in preds:
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                p.append(text)
    model.train()
    predictions = []
    output_dir = "../bleu/"
    with open(os.path.join(output_dir, "dev.output"), 'w') as f, open(
            os.path.join(output_dir, "dev.gold"), 'w') as f1:
        for ref, gold in zip(p, bleu_eval_examples):
            predictions.append(str(gold.idx) + '\t' + ref)
            f.write(str(gold.idx) + '\t' + ref + '\n')
            f1.write(str(gold.idx) + '\t' + gold.target + '\n')

    (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(output_dir, "dev.gold"))
    dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
    logger.info("  " + "*" * 20)

    results = {'eval_ppl': eval_ppl,
                'eval_bleu': dev_bleu
               }
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
def distill(tokenizer, args, map_functionList, eval=False, surrogate=False):

    set_seed(args.seed)

    dev_best_accs = []
    dev_best_f1s = []
    dev_best_pres = []
    dev_best_recs = []
    # teacher model
    config=RobertaConfig.from_pretrained('/home/wuruifeng/data/models/codebert-base')
    encoder = RobertaModel(config=config)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    teacher_model =TeacherSeq2Seq(encoder=encoder, decoder=decoder, config=config,
                    beam_size=args.beam_size, max_length=args.max_target_length,
                    sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
    teacher_model.load_state_dict(torch.load(args.teacher_model))

    teacher_model.to(args.device)


    for map_function in map_functionList:
        config=RobertaConfig.from_pretrained(args.student_model)
        encoder = RobertaModel(config=config)
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        student_model =StudentSeq2Seq(encoder=encoder, decoder=decoder, config=config,
                        beam_size=args.beam_size, max_length=args.max_target_length,
                        sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
        student_model.to(args.device)

        if not eval:
            train_examples = read_examples(args.train_filename)
            train_features = convert_examples_to_features(train_examples, tokenizer,args,stage='train')
            all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
            all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
            all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)    
            train_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)

            train_sampler = RandomSampler(train_data)

            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)


            eval_examples = read_examples(args.eval_data_file)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='dev')
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
            all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
            all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            eval_examples = read_examples(args.eval_data_file)
            eval_examples = random.sample(eval_examples, min(1000, len(eval_examples)))
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
            bleu_eval_dataloader = TensorDataset(all_source_ids, all_source_mask)
            bleu_eval_examples = eval_examples

            dev_best_outcomes = train(student_model, teacher_model, map_function, train_dataloader, eval_dataloader,bleu_eval_dataloader,bleu_eval_examples,
                                 args.hid_epoches,
                                 args.pred_epoches, args.hid_learning_rate, args.pred_learning_rate, args.temperature,
                                 args.device, args.loss_function,surrogate)
           

            dev_best_accs.append(dev_best_outcomes["best_bleu"])
            dev_best_f1s.append(dev_best_outcomes["best_ppl"])


        else:
            model_dir = os.path.join("../checkpoints", "Avatar", "model.bin")
            student_model.load_state_dict(torch.load(model_dir))
            test_dataset = TextDataset(tokenizer, args, file_path=args.eval_data_file)
            test_sampler = SequentialSampler(test_dataset)
            test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size,
                                         num_workers=8,
                                         pin_memory=True)

            student_model.to(args.device)
            test_results = evaluate(student_model, args.device, test_dataloader)
            logger.info(
                "Test Acc: {0}, Test Precision: {1}, Test Recall: {2}, Test F1: {3}".format(test_results["eval_acc"],
                                                                                            test_results[
                                                                                                "eval_precision"],
                                                                                            test_results["eval_recall"],
                                                                                            test_results["eval_f1"],
                                                                                            test_results[
                                                                                                "inference_time"]))

    return dev_best_accs, dev_best_f1s, dev_best_pres, dev_best_recs



if __name__ == "__main__":
    tokenizer = None
    args = None
    map_function = [1,2,3,4,5,6]
    map_functionList = [map_function]
    dev_best_bleu=distill(tokenizer, args, [0, 1, 2], eval=False, surrogate=True)

    print(dev_best_bleu)


