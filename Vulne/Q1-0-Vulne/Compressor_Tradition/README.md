```
CUDA_VISIBLE_DEVICES=3 nohup python distill.py \
    --train_data_file=/data/wuruifeng/data/Vulnerability-Detection/data/label_train.jsonl \
    --eval_data_file=/data/wuruifeng/data/Vulnerability-Detection/data/valid.jsonl \
    --output_dir=/data/wuruifeng/result/Vulnerability-Detection/Compressor \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 123456 \
    >tradition_dist.log 2>&1 &
```