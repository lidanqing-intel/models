#!/bin/bash
DATASET_DIR=/home/wojtuss/repos/PaddlePaddle/data/Transformer/gen_data/
MODEL_DIR=/home/wojtuss/repos/PaddlePaddle/data/Transformer/base_model/
# MODEL_DIR=/home/wojtuss/repos/PaddlePaddle/data/Transformer/big_model/
SAVE_MODEL_DIR=/home/wojtuss/repos/PaddlePaddle/data/Transformer/my_base_model_mem-opt_py-reader_10_INT64_3
cd ..
python infer.py \
        --src_vocab_fpath $DATASET_DIR/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
        --trg_vocab_fpath $DATASET_DIR/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
	--special_token '<s>' '<e>' '<unk>' \
	--test_file_pattern $DATASET_DIR/wmt16_ende_data_bpe/newstest2016.tok.bpe.32000.en-de \
	--token_delimiter ' ' \
        --batch_size 32 \
	--save_model_dir $SAVE_MODEL_DIR \
	--use_mem_opt True \
        --use_py_reader False \
	model_path $MODEL_DIR/iter_100000.infer.model/ \
	beam_size 4 \
	max_out_len 255

cd -
