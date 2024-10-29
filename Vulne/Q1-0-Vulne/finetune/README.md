Finetune CodeBERT:
```
CUDA_VISIBLE_DEVICES=1 nohup python finetune.py \
--do_train \
--train_data_file=/data/wuruifeng/data/Vulnerability-Detection/data/label_train.jsonl \
--eval_data_file=/data/wuruifeng/data/Vulnerability-Detection/data/valid.jsonl \
--output_dir=/data/wuruifeng/result/Vulnerability-Detection/Finetune-renew \
--epoch 10 \
--block_size 400 \
--train_batch_size 16 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123456 \
>finetune.log 2>&1 &
```
