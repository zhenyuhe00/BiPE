
batch_size=8
DATA_DIR=/mnt/bn/hzy-data-all/pile_pg19_tmp
MODEL=/mnt/bn/hzy-data-all/output_pile/ape1_sent_rope/step_490000

for block_size in 6144
do
    python3 eval.py\
        --dataset_cache_dir /mnt/bn/hzy-data-all/pile_pg19\
        --block_size $block_size\
        --tokenizer_name llama_tokenizer\
        --per_device_eval_batch_size $batch_size \
        --model_name_or_path $MODEL
done
