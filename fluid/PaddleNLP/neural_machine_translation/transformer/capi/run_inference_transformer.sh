export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=12
cd build
#cgdb --args ./infer_neural_machine_translation \
./infer_neural_machine_translation \
  --infer_model=/home/li/models/iter_100000.infer.model \
  --all_vocab_fpath=/home/li/data/wmt16_ende_data_bpe_clean/vocab_all.bpe.32000 \
  --test_file_path=/home/li/data/wmt16_ende_data_bpe_clean/newstest2016.tok.bpe.32000.en-de \
  --batch_size=8 \
  --use_mkldnn=true \
  --skip_passes=false \
  --enable_graphviz=1 \
  --paddle_num_threads=12 \
  --profile=true \
  --beam_size=4 \
  --max_out_len=255 \
  --iterations=380  \
  --skip_batch_num=0 \
  --output_file=./output_file_bs1.txt \

sed -r 's/(@@ )|(@@ ?$)//g' output_file_bs1.txt > output.tok.txt
perl /home/li/data/mosesdecoder/scripts/generic/multi-bleu.perl  /home/li/data/wmt16_ende_data_bpe_clean/newstest2016.tok.de < output.tok.txt

cd -
