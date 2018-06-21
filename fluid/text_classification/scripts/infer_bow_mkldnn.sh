#!/bin/bash
time python ../infer.py \
	--device CPU \
	--model_path bow_model_mkldnn/epoch0 \
	--num_passes 100 \
	--profile
