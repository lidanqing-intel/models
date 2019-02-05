
// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gflags/gflags.h>
#include <stdio.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include "data_reader.h"
#include "paddle/fluid/inference/paddle_inference_api.h"
#include "paddle/fluid/platform/profiler.h"
#include "stats.h"

DEFINE_string(infer_model, "", "Directory of the inference model.");
DEFINE_string(all_vocab_fpath,
              "",
              "Path to a vocabulary file with both languages.");
DEFINE_string(test_file_path, "", "The test data file with sentences.");
DEFINE_string(output_file, "out_file.txt", "A file for the model output.");
DEFINE_int32(batch_size,
             8,
             "The number of examples in one run for sequence generation.");

DEFINE_int32(iterations, 5, "How many times to repeat run.");
DEFINE_int32(skip_batch_num, 0, "How many minibatches to skip in statistics.");
DEFINE_bool(use_mkldnn, false, "Use MKL-DNN.");
DEFINE_bool(use_fake_data, false, "Use fake data or not.");
DEFINE_bool(skip_passes, false, "Skip running passes.");
DEFINE_bool(enable_graphviz,
            false,
            "Enable graphviz to get .dot files with data flow graphs.");
DEFINE_bool(profile, false, "Turn on profiler for fluid");
DEFINE_int32(paddle_num_threads,
             1,
             "Number of threads for each paddle instance.");
DEFINE_int32(beam_size, 4, "Search width for Beam Search algorithm.");
DEFINE_int32(
    max_out_len,
    255,
    "The maximum depth(translation length) for Beam Search algorithm.");
DEFINE_int32(n_head, 8, "Number of attention heads.");

namespace {
class Timer {
public:
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point startu;

  void tic() { start = std::chrono::high_resolution_clock::now(); }
  double toc() {
    startu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(startu -
                                                                  start);
    double used_time_ms = static_cast<double>(time_span.count()) * 1000.0;
    return used_time_ms;
  }
};

}  // namespace
namespace paddle {
void PrintOutput(const std::vector<paddle::PaddleTensor>& output,
                 const std::string& out_file,
                 std::unique_ptr<DataReader>& reader) {
  if (output.size() == 0)
    throw std::invalid_argument("PrintOutput: output vector is empty.");

  if (output[0].dtype != paddle::PaddleDType::INT64) {
    throw std::invalid_argument("PrintOutput: output is of a wrong type.");
  }

  int64_t* ids_data = static_cast<int64_t*>(output[0].data.data());
  auto ids_lod = output[0].lod;
  auto ids_shape = output[0].shape;

  std::ofstream ofile(out_file, std::fstream::out | std::fstream::app);
  if (!ofile.is_open())
    throw std::invalid_argument("PrintOutput: cannot open the output file");
  for (size_t i = 0; i < ids_lod[0].size() - 1; ++i) {
    auto start = ids_lod[0][i];
    auto sub_start = ids_lod[1][start];
    auto sub_end = ids_lod[1][start + 1];
    auto data_start = ids_data + sub_start;
    auto data_end = ids_data + sub_end;
    std::vector<int64_t> indices(data_start, data_end);
    std::string sentence = reader->convert_to_sentence(indices);
    // std::cout << sentence << std::endl;
    ofile << sentence << std::endl;
  }
  ofile.close();
}

void InitializeReader(std::unique_ptr<DataReader>& reader) {
  reader.reset(new DataReader(
      FLAGS_all_vocab_fpath, FLAGS_test_file_path, FLAGS_batch_size));
}

template <typename T>
void copy_vector_of_vector(const std::vector<std::vector<T>>& src_v_v,
                           T* dst_array_ptr) {
  auto* dst_ptr = dst_array_ptr;
  for (auto v : src_v_v) {
    std::copy(v.begin(), v.end(), dst_ptr);
    dst_ptr += v.size();
  }
}

// bool ReadNextBatch(PaddleTensor& src_word_tensor,
// PaddleTensor& src_pos_tensor,
// PaddleTensor& src_slf_attn_bias_tensor,
// PaddleTensor& trg_word_tensor,
// PaddleTensor& init_score_tensor,
// PaddleTensor& init_idx_tensor,
// PaddleTensor& trg_src_attn_bias_tensor,
// std::unique_ptr<DataReader>& reader) {
bool ReadNextBatch(PaddleTensor& trg_word_tensor,
                   PaddleTensor& src_word_tensor,
                   PaddleTensor& init_idx_tensor,
                   PaddleTensor& src_slf_attn_bias_tensor,
                   PaddleTensor& src_pos_tensor,
                   PaddleTensor& trg_src_attn_bias_tensor,
                   PaddleTensor& init_score_tensor,
                   std::unique_ptr<DataReader>& reader) {
  std::vector<std::vector<int64_t>> inst_data;
  std::vector<std::vector<int64_t>> inst_pos;
  std::vector<std::vector<float>> slf_attn_bias_data;
  int max_length = 0;

  bool read_full_batch = reader->NextBatch(
      inst_data, inst_pos, slf_attn_bias_data, max_length, FLAGS_batch_size);

  // pad_batch_data
  if (read_full_batch == false) {
    return false;  // we didn't read full batch. Stop or throw.
  }

  src_word_tensor.name = "src_word";
  src_pos_tensor.name = "src_pos";
  src_slf_attn_bias_tensor.name = "src_slf_attn_bias";
  trg_word_tensor.name = "trg_word";
  init_score_tensor.name = "init_score";
  init_idx_tensor.name = "init_idx";
  trg_src_attn_bias_tensor.name = "trg_src_attn_bias";

  src_word_tensor.shape = {FLAGS_batch_size, max_length, 1};
  src_word_tensor.data.Resize(FLAGS_batch_size * max_length * sizeof(int64_t));
  src_word_tensor.dtype = PaddleDType::INT64;
  src_word_tensor.lod.clear();

  src_pos_tensor.shape = {FLAGS_batch_size, max_length, 1};
  src_pos_tensor.data.Resize(FLAGS_batch_size * max_length * sizeof(int64_t));
  src_pos_tensor.lod.clear();
  src_pos_tensor.dtype = PaddleDType::INT64;

  trg_src_attn_bias_tensor.shape = {
      FLAGS_batch_size, FLAGS_n_head, 1, max_length};
  trg_src_attn_bias_tensor.data.Resize(FLAGS_batch_size * FLAGS_n_head * 1 *
                                       max_length * sizeof(float));
  trg_src_attn_bias_tensor.lod.clear();
  trg_src_attn_bias_tensor.dtype = PaddleDType::FLOAT32;

  src_slf_attn_bias_tensor.shape = {
      FLAGS_batch_size, FLAGS_n_head, max_length, max_length};
  src_slf_attn_bias_tensor.data.Resize(FLAGS_batch_size * FLAGS_n_head *
                                       max_length * max_length * sizeof(float));
  src_slf_attn_bias_tensor.lod.clear();

  src_slf_attn_bias_tensor.dtype = PaddleDType::FLOAT32;

  trg_word_tensor.shape = {FLAGS_batch_size, 1, 1};
  trg_word_tensor.data.Resize(FLAGS_batch_size * 1 * 1 * sizeof(int64_t));

  trg_word_tensor.dtype = PaddleDType::INT64;
  trg_word_tensor.lod.clear();
  std::vector<size_t> tmplod;
  for (int i = 0; i <= FLAGS_batch_size; i++) {
    tmplod.push_back(i);
  }
  trg_word_tensor.lod = {tmplod, tmplod};
  int64_t* trg_word_array = static_cast<int64_t*>(trg_word_tensor.data.data());
  for (int i = 0; i < FLAGS_batch_size; i++) {
    *trg_word_array++ = reader->bos_idx;
  }

  init_idx_tensor.shape = {FLAGS_batch_size};
  init_idx_tensor.data.Resize(FLAGS_batch_size * sizeof(int64_t));
  init_idx_tensor.lod.clear();
  init_idx_tensor.dtype = PaddleDType::INT64;
  int64_t* init_idx_array = static_cast<int64_t*>(init_idx_tensor.data.data());
  for (int64_t i = 0; i < FLAGS_batch_size; *init_idx_array++ = i++)
    ;

  init_score_tensor.shape = {FLAGS_batch_size, 1};
  init_score_tensor.data.Resize(FLAGS_batch_size * 1 * sizeof(float));
  init_score_tensor.dtype = PaddleDType::FLOAT32;
  float* init_score_array = static_cast<float*>(init_score_tensor.data.data());
  for (int i = 0; i < FLAGS_batch_size; i++) {
    *init_score_array++ = 0;
  }
  init_score_tensor.lod = {tmplod, tmplod};

  int64_t* src_word_array = static_cast<int64_t*>(src_word_tensor.data.data());
  int64_t* src_pos_array = static_cast<int64_t*>(src_pos_tensor.data.data());

  float* trg_src_attn_bias_array =
      static_cast<float*>(trg_src_attn_bias_tensor.data.data());
  for (int i = 0; i < FLAGS_batch_size; i++) {
    for (int j = 0; j < reader->n_head; j++) {
      std::copy(slf_attn_bias_data[i].begin(),
                slf_attn_bias_data[i].end(),
                trg_src_attn_bias_array + i * reader->n_head * max_length +
                    j * max_length);
    }
  }

  float* src_slf_attn_bias_array =
      static_cast<float*>(src_slf_attn_bias_tensor.data.data());
  // tile, batch_size*n_head*max_length*max_length
  for (int i = 0; i < FLAGS_batch_size; i++) {
    for (int j = 0; j < reader->n_head * max_length; j++) {
      std::copy(slf_attn_bias_data[i].begin(),
                slf_attn_bias_data[i].end(),
                src_slf_attn_bias_array +
                    i * reader->n_head * max_length * max_length +
                    j * max_length);
    }
  }

  copy_vector_of_vector(inst_data, src_word_array);
  copy_vector_of_vector(inst_pos, src_pos_array);
  return true;
}


#define PRINT_OPTION(a)                               \
  do {                                                \
    std::cout << #a ": " << (FLAGS_##a) << std::endl; \
  } while (false)

void PrintInfo() {
  std::cout << std::endl
            << "--- Used Parameters: -----------------" << std::endl;
  PRINT_OPTION(all_vocab_fpath);
  PRINT_OPTION(batch_size);
  PRINT_OPTION(beam_size);
  PRINT_OPTION(enable_graphviz);
  PRINT_OPTION(infer_model);
  PRINT_OPTION(max_out_len);
  PRINT_OPTION(skip_batch_num);
  PRINT_OPTION(iterations);
  PRINT_OPTION(output_file);
  PRINT_OPTION(paddle_num_threads);
  PRINT_OPTION(profile);
  PRINT_OPTION(skip_passes);
  PRINT_OPTION(test_file_path);
  PRINT_OPTION(use_mkldnn);
  std::cout << "--------------------------------------" << std::endl;
}

void PrepareConfig(AnalysisConfig& config) {
  config.SetModel(FLAGS_infer_model);
  config.DisableGpu();
  // config.SwitchIrOptim(!FLAGS_skip_passes);
  config.SwitchIrOptim(false);
  config.SwitchSpecifyInputNames(false);
  config.SetCpuMathLibraryNumThreads(FLAGS_paddle_num_threads);
  if (FLAGS_use_mkldnn) config.EnableMKLDNN();

  // remove all passes so that we can add them in correct order
  for (int i = config.pass_builder()->AllPasses().size() - 1; i >= 0; i--)
    config.pass_builder()->DeletePass(i);

  // add passes
  config.pass_builder()->AppendPass("infer_clean_graph_pass");
  if (FLAGS_use_mkldnn) {
    // add passes to execute with MKL-DNN
    config.pass_builder()->AppendPass("mkldnn_placement_pass");
    config.pass_builder()->AppendPass("is_test_pass");
    config.pass_builder()->AppendPass("depthwise_conv_mkldnn_pass");
    config.pass_builder()->AppendPass("conv_bn_fuse_pass");
    config.pass_builder()->AppendPass("conv_eltwiseadd_bn_fuse_pass");
    config.pass_builder()->AppendPass("conv_bias_mkldnn_fuse_pass");
    config.pass_builder()->AppendPass("conv_elementwise_add_mkldnn_fuse_pass");
    config.pass_builder()->AppendPass("conv_relu_mkldnn_fuse_pass");
    config.pass_builder()->AppendPass("fc_fuse_pass");
  } else {
    // add passes to execute keeping the order - without MKL-DNN
    config.pass_builder()->AppendPass("conv_bn_fuse_pass");
    config.pass_builder()->AppendPass("fc_fuse_pass");
  }

  if (FLAGS_enable_graphviz) config.pass_builder()->TurnOnDebug();
}

void Main() {
  PrintInfo();
  std::ofstream file{FLAGS_output_file};
  if (FLAGS_batch_size <= 0)
    throw std::invalid_argument(
        "The batch_size option is less than or equal to 0.");

  if (FLAGS_iterations <= 0)
    throw std::invalid_argument(
        "The iterations option is less than or equal to 0.");

  if (FLAGS_skip_batch_num < 0)
    throw std::invalid_argument("The skip_batch_num option is less than 0.");

  if (FLAGS_paddle_num_threads <= 0)
    throw std::invalid_argument(
        "The paddle_num_threads option is less than or equal to 0.");

  if (FLAGS_beam_size <= 0)
    throw std::invalid_argument(
        "The beam_size option is less than or equal to 0.");

  if (FLAGS_n_head <= 0)
    throw std::invalid_argument(
        "The n_head option is less than or equal to 0.");

  if (FLAGS_max_out_len <= 0)
    throw std::invalid_argument(
        "The max_out_len option is less than or equal to 0.");


  struct stat sb;
  if (stat(FLAGS_infer_model.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode)) {
    throw std::invalid_argument(
        "The inference model directory does not exist.");
  }

  if (stat(FLAGS_all_vocab_fpath.c_str(), &sb) != 0 || !S_ISREG(sb.st_mode)) {
    throw std::invalid_argument("The vocabulary file does not exist.");
  }

  if (stat(FLAGS_test_file_path.c_str(), &sb) != 0 || !S_ISREG(sb.st_mode)) {
    throw std::invalid_argument("The test data file does not exist.");
  }

  if (stat(FLAGS_output_file.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)) {
    std::cout << "Warning: The output file " + FLAGS_output_file +
                     " already exists and it will be used in append mode!\n";
  }

  paddle::Stats stats(FLAGS_batch_size, FLAGS_skip_batch_num);
  std::unique_ptr<DataReader> reader;

  InitializeReader(reader);

  std::vector<PaddleTensor> input(7);
  bool read_full_batch = ReadNextBatch(input[0],
                                       input[1],
                                       input[2],
                                       input[3],
                                       input[4],
                                       input[5],
                                       input[6],
                                       reader);
  if (read_full_batch == false)
    throw std::runtime_error("File contains less than one batch of data!\n");

  // configure predictor
  AnalysisConfig config;
  PrepareConfig(config);

  auto predictor =
      CreatePaddlePredictor<AnalysisConfig, PaddleEngineKind::kAnalysis>(
          config);

  if (FLAGS_profile) {
    auto pf_state = paddle::platform::ProfilerState::kCPU;
    paddle::platform::EnableProfiler(pf_state);
  }

  // define output and timer
  std::vector<PaddleTensor> output_slots;
  Timer timer, timer_total;

  // run prediction
  for (int i = 0; i < FLAGS_iterations + FLAGS_skip_batch_num; i++) {
    if (i == FLAGS_skip_batch_num) {
      timer_total.tic();
      if (FLAGS_profile) {
        paddle::platform::ResetProfiler();
      }
    }

    // read next batch of data
    if (i > 0 && !FLAGS_use_fake_data) {
      if (!ReadNextBatch(input[0],
                         input[1],
                         input[2],
                         input[3],
                         input[4],
                         input[5],
                         input[6],
                         reader)) {
        std::cout << "\n Less than one batch of data. Stopping." << std::endl;
        break;
      }
    }
    timer.tic();
    if (!predictor->Run(input, &output_slots))
      throw std::runtime_error("Prediction failed.");

    double batch_time = timer.toc() / 1000;
    std::string prefix = i < FLAGS_skip_batch_num ? " warm up batch num "
                                                  : " profiling batch num ";
    std::string midfix = ", batch size " + std::to_string(FLAGS_batch_size);
    std::cout << "\n+++" << prefix << i << midfix << ", batch time "
              << batch_time << "+++";
    PrintOutput(output_slots, FLAGS_output_file, reader);
    stats.GatherTime(batch_time, i);
  }

  double total_samples = FLAGS_iterations * FLAGS_batch_size;
  double total_time = timer_total.toc() / 1000;
  stats.Postprocess(total_time, total_samples);

  // disable profiler
  if (FLAGS_profile) {
    paddle::platform::DisableProfiler(paddle::platform::EventSortingKey::kTotal,
                                      "/tmp/profiler");
  }
}

}  // namespace paddle

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  try {
    paddle::Main();
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return 0;
}
