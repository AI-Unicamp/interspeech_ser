#!/bin/bash

#SSL model version
ssl_type=wavlm-large

#Pooling method
pool_type=AttentiveStatisticsPooling

# Train/Eval
for seed in 7; do
    python train_eval_files/train_dim_ser.py \
        --seed=${seed} \
        --ssl_type=${ssl_type} \
        --batch_size=32 \
        --accumulation_steps=4 \
        --lr=1e-5 \
        --epochs=20 \
        --pooling_type=${pool_type} \
        --model_path=model/dim_ser/${seed} || exit 0;

    # Evaluation on Test3 and save results using format required by challenge
    python train_eval_files/eval_dim_ser_test3.py \
        --ssl_type=${ssl_type} \
        --pooling_type=${pool_type} \
        --model_path=model/dim_ser/${seed}  \
        --store_path=result/dim_ser/${seed}.txt || exit 0;

    # General evaluation code for sets with labels
    python train_eval_files/eval_dim_ser.py \
        --ssl_type=${ssl_type} \
        --pooling_type=${pool_type} \
        --model_path=model/dim_ser/${seed}  \
        --store_path=result/dim_ser/${seed}.txt || exit 0;
    
done
