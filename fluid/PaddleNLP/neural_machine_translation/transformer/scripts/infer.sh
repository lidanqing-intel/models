export FLAGS_use_mkldnn=1
#export OMP_NUM_THREADS=1

#python -m pdb ../infer.py \
python ../infer.py \
  --src_vocab_fpath ~/data/wmt16_ende_data_bpe_clean/vocab_all.bpe.32000 \
  --trg_vocab_fpath ~/data/wmt16_ende_data_bpe_clean/vocab_all.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --test_file_pattern ~/data/wmt16_ende_data_bpe_clean/newstest2016.tok.bpe.32000.en-de \
  --token_delimiter ' ' \
  --batch_size 8 \
  --save_model_dir "/home/li/save_iter1000/" \
  --use_mem_opt True \
  --use_py_reader False \
  model_path ~/models/iter_100000.infer.model \
  beam_size 4 \
  max_out_len 255 \
  use_gpu False

