# 

```
CUDA_VISIBLE_DEVICES=2  nohup python finetune.py \
	--pred_model_dir=/data/wuruifeng/result/Vulnerability-Detection/Finetune/finetune \
    --eval_data_file=/data/wuruifeng/data/Vulnerability-Detection/data/test.jsonl \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 123456 \
     >baseline_eval_dist.log 2>&1 &
```


