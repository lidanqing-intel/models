#!/bin/bash
# train model resnet_cifar10 using dataset cifar10
time python ../train_resnet.py \
	--device CPU \
	--batch_size 32 \
	--skip_batch_num 0 \
	--pass_num 1 \
	--iterations 1 \
	--model resnet_cifar10 \
	--data_set cifar10 \
	--skip_test \
	--save_model \
	--save_model_path ../models/resnet_cifar10_cifar10