/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "main_functions.h"

#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"

#include "examples/luna_classification/classification_model_01.h"
// #include "examples/luna_classification/classification_model_01_float_fallback.h"
// #include "examples/luna_classification/classification_model_01_no_quantization.h"

#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
int8_t* image_data = nullptr;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 136 * 1024; // TODO FIGURE THIS NUMBER OUT
#if (defined(__GNUC__) || defined(__GNUG__)) && !defined (__CCAC__)
static uint8_t tensor_arena[kTensorArenaSize] __attribute__((section(".tensor_arena")));
#else
#pragma Bss(".tensor_arena")
static uint8_t tensor_arena[kTensorArenaSize];
#pragma Bss()
#endif // if defined (_GNUC_) && !defined (_CCAC_)
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  tflite::InitializeTarget();

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(classification_model_01_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  // static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  // micro_op_resolver.AddAveragePool2D();
  // micro_op_resolver.AddConv2D();
  // micro_op_resolver.AddDepthwiseConv2D();
  // micro_op_resolver.AddReshape();
  // micro_op_resolver.AddSoftmax();

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);
  
  image_data = new int8_t[kNumCols*kNumRows*kNumChannels];
}

// The name of this function is important for Arduino compatibility.
void loop() {
  // Get image from provider.
  if (kTfLiteOk != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels,
                            image_data)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
  }

  if (input->type == TfLiteType::kTfLiteUInt8) {
    void* quanti_param = input->quantization.params;
    assert(input->quantization.type == TfLiteQuantizationType::kTfLiteAffineQuantization);
    TfLiteAffineQuantization* quanti_affine_param = static_cast<TfLiteAffineQuantization*>(quanti_param);
    for (uint i = 0; i < (kNumCols*kNumRows); i+=3) {
      float r = (float) image_data[i+0];
      float g = (float) image_data[i+1];
      float b = (float) image_data[i+2];
      // ImageNet Normalization
      r = ((r+128)/255.0 -0.406) / 0.225;
      g = ((g+128)/255.0 -0.456) / 0.224;
      b = ((b+128)/255.0 -0.485) / 0.229;
      // Quantization normalization
      r = (r / quanti_affine_param->scale->data[0]) + quanti_affine_param->zero_point->data[0];
      g = (g / quanti_affine_param->scale->data[1]) + quanti_affine_param->zero_point->data[1];
      b = (b / quanti_affine_param->scale->data[2]) + quanti_affine_param->zero_point->data[2];
      input->data.uint8[i+0] = (uint8_t)b;
      input->data.uint8[i+1] = (uint8_t)g;
      input->data.uint8[i+2] = (uint8_t)r;
    }
  } else if (input->type == TfLiteType::kTfLiteFloat32) {
    for (uint i = 0; i < (kNumCols*kNumRows); i+=3) {
      float r = (float) image_data[i+0];
      float g = (float) image_data[i+1];
      float b = (float) image_data[i+2];
      // ImageNet Normalization
      r = ((r+128)/255.0 -0.406) / 0.225;
      g = ((g+128)/255.0 -0.456) / 0.224;
      b = ((b+128)/255.0 -0.485) / 0.229;
      input->data.f[i+0] = b;
      input->data.f[i+1] = g;
      input->data.f[i+2] = r;
    }
  } else {
    assert(false);
  }

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
  }

  TfLiteTensor* output = interpreter->output(0);

  // Process the inference results.
  int8_t road_score = output->data.uint8[kRoadIndex];
  int8_t bike_score = output->data.uint8[kBikeIndexIndex];
  RespondToClassification(error_reporter, road_score, bike_score);
}
