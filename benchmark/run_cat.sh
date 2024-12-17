#!/bin/bash

#SSL model version
ssl_type=wavlm-large

#Pooling method
pool_type=AttentiveStatisticsPooling

# Train/Eval
for seed in 7; do
    python train_eval_files/train_cat_ser.py \
        --seed=${seed} \
        --ssl_type=${ssl_type} \
        --batch_size=32 \
        --accumulation_steps=4 \
        --lr=1e-5 \
        --epochs=20 \
        --pooling_type=${pool_type} \
        --model_path=model/cat_ser/${seed} || exit 0;
    
    python train_eval_files/eval_cat_ser.py \
        --ssl_type=${ssl_type} \
        --pooling_type=${pool_type} \
        --model_path=model/cat_ser/${seed}  \
        --store_path=result/cat_ser/${seed}.txt || exit 0;

done
