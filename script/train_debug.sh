OUTPUT_DIR=/mnt/bn/hzy-data-all/output_pile/tmp  # path to save checkpoints and tensorboard
DATA_DIR=/mnt/bn/hzy-data-all/wikitext-130-raw-v1  # path to load data
CONFIG_NAME=config/rope.json # choose from [config/bipe_rope.json, config/bipe_alibi.json, config/rope.json, config/alibi.json]


deepspeed --master_port 25401 train.py \
    --dataset_cache_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --config_name config/bipe_rope.json \
    --max_steps 1000000 \
    --warmup_steps 10000 \
    --lr_scheduler_type polynomial \
    --save_steps 10000 \
    --eval_steps 10000 \
    --logging_steps 50 \
    --weight_decay 0.01 \
    --learning_rate 1e-4 \
    --model_max_position_embeddings 1024 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --do_train True \
    --do_eval True \
    --do_predict True \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --load_best_model_at_end True \
    --report_to "tensorboard" \
    --gradient_checkpointing False \
    --fp16 True