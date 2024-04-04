
[ -z "${batch_size}" ] && batch_size=8
[ -z "${DATA_DIR}" ] && DATA_DIR=./data # data path
[ -z "${MODEL}" ] && MODEL=./bipe_rope # checkpoint path

for block_size in 1024 2048 3072 4096 5120 6144
do
    python3 eval.py\
        --dataset_cache_dir ${DATA_DIR}_pg19\
        --block_size $block_size\
        --tokenizer_name llama_tokenizer\
        --per_device_eval_batch_size $batch_size \
        --model_name_or_path $MODEL
    python3 eval.py\
        --dataset_cache_dir ${DATA_DIR}_arxiv\
        --block_size $block_size\
        --tokenizer_name llama_tokenizer\
        --per_device_eval_batch_size $batch_size \
        --model_name_or_path $MODEL
    python3 eval.py\
        --dataset_cache_dir ${DATA_DIR}_github\
        --block_size $block_size\
        --tokenizer_name llama_tokenizer\
        --per_device_eval_batch_size $batch_size \
        --model_name_or_path $MODEL
done
