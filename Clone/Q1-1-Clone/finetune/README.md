```
CUDA_VISIBLE_DEVICES=3 python finetune.py \
    --train_data_file=/data/wuruifeng/data/Clone-Detection/data/label_train.txt \
    --eval_data_file=/data/wuruifeng/data/Clone-Detection/data/valid_sampled.txt \
    --output_dir=/data/wuruifeng/result/Clone-Detection/Finetune/ \
    --epochs 10 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 100 \
    --seed 123456
```