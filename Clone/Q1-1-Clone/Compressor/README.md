```
CUDA_VISIBLE_DEVICES=4 python3 distill.py \
    --student_model=/data/wuruifeng/models/4layer_config \
    --teacher_model= \
    --train_data_file=/data/wuruifeng/data/Clone-Detection/data/label_train.txt \
    --eval_data_file=/data/wuruifeng/data/Clone-Detection/data/valid_sampled.txt \
    --output_dir /data/wuruifeng/result/Clone-Detection/4layer/ \
    --train_batch_size 16 \
    --eval_batch_size 100 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seed 123456
```
