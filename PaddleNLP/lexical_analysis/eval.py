# -*- coding: UTF-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import time
import sys
import numpy as np
import paddle.fluid as fluid
import paddle
from paddle.fluid import profiler
import utils
import reader
import creator
sys.path.append('../shared_modules/models/')
from model_check import check_cuda
from model_check import check_version

parser = argparse.ArgumentParser(__doc__)
# 1. model parameters
model_g = utils.ArgumentGroup(parser, "model", "model configuration")
model_g.add_arg("word_emb_dim", int, 128,
                "The dimension in which a word is embedded.")
model_g.add_arg("grnn_hidden_dim", int, 128,
                "The number of hidden nodes in the GRNN layer.")
model_g.add_arg("bigru_num", int, 2,
                "The number of bi_gru layers in the network.")
model_g.add_arg("use_cuda", bool, False, "If set, use GPU for training.")

# 2. data parameters
data_g = utils.ArgumentGroup(parser, "data", "data paths")
data_g.add_arg("word_dict_path", str, "./conf/word.dic",
               "The path of the word dictionary.")
data_g.add_arg("label_dict_path", str, "./conf/tag.dic",
               "The path of the label dictionary.")
data_g.add_arg("word_rep_dict_path", str, "./conf/q2b.dic",
               "The path of the word replacement Dictionary.")
data_g.add_arg("test_data", str, "./data/test.tsv",
               "The folder where the training data is located.")
data_g.add_arg("init_checkpoint", str, "./model_baseline", "Path to init model")
data_g.add_arg(
    "batch_size", int, 1,
    "The number of sequences contained in a mini-batch, "
    "or the maximum number of tokens (include paddings) contained in a mini-batch."
)
model_g.add_arg("eval_save_dir", str, "./GRU_eval_model_v3",
                "The number of hidden nodes in the GRNN layer.")


def do_eval(args):
    dataset = reader.Dataset(args)

    test_program = fluid.Program()
    with fluid.program_guard(test_program, fluid.default_startup_program()):
        with fluid.unique_name.guard():
            test_ret = creator.create_model(
                args, dataset.vocab_size, dataset.num_labels, mode='test')
    test_program = test_program.clone(for_test=True)

    # init executor
    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
    else:
        place = fluid.CPUPlace()

    pyreader = creator.create_pyreader(
        args,
        file_name=args.test_data,
        feed_list=test_ret['feed_list'],
        place=place,
        model='lac',
        reader=dataset,
        mode='test')
    profiler.start_profiler("All")
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # load model
    utils.init_checkpoint(exe, args.init_checkpoint, test_program)

    test_process(
        exe=exe, program=test_program, reader=pyreader, test_ret=test_ret)

    profiler.stop_profiler("total", "/tmp/gru_profile")

def test_process(exe, program, reader, test_ret):
    """
    the function to execute the infer process
    :param exe: the fluid Executor
    :param program: the infer_program
    :param reader: data reader
    :return: the list of prediction result
    """
    test_ret["chunk_evaluator"].reset()

    fluid.io.save_inference_model(
        args.eval_save_dir,
        ['words','targets'],
        [ test_ret["num_infer_chunks"], test_ret["num_label_chunks"], test_ret["num_correct_chunks"] ],
        exe,
        main_program=program,
        model_filename=None,
        params_filename=None)
    x = fluid.framework._get_var('embedding_0.tmp_0', program)
    lods = []
    words = []
    targets = []
    sum_words = 0
    sum_sentences = 0
    i = 0
    start_time = time.time()
    for data in reader():
        print(len(data[0]['words'].lod()[0]))
        print(data[0]['words'])
        # exit(0)
        new_lod = data[0]['words'].lod()[0][1]
        # print("new lod is ", new_lod)
        new_words = np.array(data[0]['words'])
        new_targets = np.array(data[0]['targets'])
        assert new_lod == len(new_words)
        assert new_lod == len(new_targets)
        lods.append(new_lod)
        words.extend(new_words.flatten())
        targets.extend(new_targets.flatten())
        sum_sentences = sum_sentences + 1
        sum_words = sum_words + new_lod
        nums_infer, nums_label, nums_correct = exe.run(
            program,
            fetch_list=[
                test_ret["num_infer_chunks"],
                test_ret["num_label_chunks"],
                test_ret["num_correct_chunks"],
            ],
            feed=data, )
        print("nums_infer %d, nums_label %d, nums_correct %d" % (nums_infer, nums_label, nums_correct))
        tmp=exe.run(program, feed=data, fetch_list=[x])
        print("WARNING!!!!! HERE IS THE LOOKUP TABLE OUTPUT VALUES")
        print(tmp)
        print("***************************************************")
        test_ret["chunk_evaluator"].update(nums_infer, nums_label, nums_correct)
    precision, recall, f1 = test_ret["chunk_evaluator"].eval()
    end_time = time.time()
    print("[test] P: %.5f, R: %.5f, F1: %.5f, elapsed time: %.3f s" %
          (precision, recall, f1, end_time - start_time))
    
    file1 = open("test_1022.bin","w+b")
    file1.write(np.array(int(sum_sentences)).astype('int64').tobytes())
    file1.write(np.array(int(sum_words)).astype('int64').tobytes())
    file1.write(np.array(lods).astype('uint64').tobytes())
    file1.write(np.array(words).astype('int64').tobytes())
    file1.write(np.array(targets).astype('int64').tobytes())
    file1.close()

if __name__ == '__main__':
    args = parser.parse_args()
    check_cuda(args.use_cuda)
    check_version()
    do_eval(args)
