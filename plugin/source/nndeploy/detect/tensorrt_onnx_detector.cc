#include "nndeploy/detect/tensorrt_onnx_detector.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>

#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/detect/util.h"
#include "nndeploy/device/device.h"
#include "nndeploy/inference/tensorrt/tensorrt_inference_param.h"

namespace nndeploy {
namespace detect {

base::Status TensorRtOnnxDetectorParam::serialize(
    rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
  rapidjson::Value model_path_value(
      model_path_.c_str(),
      static_cast<rapidjson::SizeType>(model_path_.length()), allocator);
  json.AddMember("model_path_", model_path_value, allocator);
  json.AddMember("input_width_", input_width_, allocator);
  json.AddMember("input_height_", input_height_, allocator);
  json.AddMember("score_threshold_", score_threshold_, allocator);
  json.AddMember("nms_threshold_", nms_threshold_, allocator);
  json.AddMember("num_classes_", num_classes_, allocator);
  json.AddMember("normalize_input_", normalize_input_, allocator);
  json.AddMember("swap_rb_", swap_rb_, allocator);
  json.AddMember("boxes_are_center_format_", boxes_are_center_format_, allocator);
  json.AddMember("has_class_probabilities_", has_class_probabilities_, allocator);
  json.AddMember("max_batch_size_", max_batch_size_, allocator);
  json.AddMember("precision_type_", static_cast<int>(precision_type_), allocator);
  json.AddMember("device_type_code_", device_type_.code_, allocator);
  json.AddMember("device_type_id_", device_type_.device_id_, allocator);

  rapidjson::Value mean_array(rapidjson::kArrayType);
  rapidjson::Value std_array(rapidjson::kArrayType);
  for (int i = 0; i < 3; ++i) {
    mean_array.PushBack(mean_[i], allocator);
    std_array.PushBack(std_[i], allocator);
  }
  json.AddMember("mean_", mean_array, allocator);
  json.AddMember("std_", std_array, allocator);
  return base::kStatusCodeOk;
}

base::Status TensorRtOnnxDetectorParam::deserialize(rapidjson::Value &json) {
  if (json.HasMember("model_path_") && json["model_path_"].IsString()) {
    model_path_ = json["model_path_"].GetString();
  }
  if (json.HasMember("input_width_") && json["input_width_"].IsInt()) {
    input_width_ = json["input_width_"].GetInt();
  }
  if (json.HasMember("input_height_") && json["input_height_"].IsInt()) {
    input_height_ = json["input_height_"].GetInt();
  }
  if (json.HasMember("score_threshold_") && json["score_threshold_"].IsFloat()) {
    score_threshold_ = json["score_threshold_"].GetFloat();
  }
  if (json.HasMember("nms_threshold_") && json["nms_threshold_"].IsFloat()) {
    nms_threshold_ = json["nms_threshold_"].GetFloat();
  }
  if (json.HasMember("num_classes_") && json["num_classes_"].IsInt()) {
    num_classes_ = json["num_classes_"].GetInt();
  }
  if (json.HasMember("normalize_input_") && json["normalize_input_"].IsBool()) {
    normalize_input_ = json["normalize_input_"].GetBool();
  }
  if (json.HasMember("swap_rb_") && json["swap_rb_"].IsBool()) {
    swap_rb_ = json["swap_rb_"].GetBool();
  }
  if (json.HasMember("boxes_are_center_format_") &&
      json["boxes_are_center_format_"].IsBool()) {
    boxes_are_center_format_ = json["boxes_are_center_format_"].GetBool();
  }
  if (json.HasMember("has_class_probabilities_") &&
      json["has_class_probabilities_"].IsBool()) {
    has_class_probabilities_ = json["has_class_probabilities_"].GetBool();
  }
  if (json.HasMember("max_batch_size_") && json["max_batch_size_"].IsInt()) {
    max_batch_size_ = json["max_batch_size_"].GetInt();
  }
  if (json.HasMember("precision_type_") && json["precision_type_"].IsInt()) {
    precision_type_ =
        static_cast<base::PrecisionType>(json["precision_type_"].GetInt());
  }
  if (json.HasMember("device_type_code_") && json["device_type_code_"].IsInt()) {
    device_type_.code_ =
        static_cast<base::DeviceTypeCode>(json["device_type_code_"].GetInt());
  }
  if (json.HasMember("device_type_id_") && json["device_type_id_"].IsInt()) {
    device_type_.device_id_ = json["device_type_id_"].GetInt();
  }
  if (json.HasMember("mean_") && json["mean_"].IsArray() &&
      json["mean_"].Size() == 3) {
    for (int i = 0; i < 3; ++i) {
      if (json["mean_"][i].IsFloat()) {
        mean_[i] = json["mean_"][i].GetFloat();
      }
    }
  }
  if (json.HasMember("std_") && json["std_"].IsArray() &&
      json["std_"].Size() == 3) {
    for (int i = 0; i < 3; ++i) {
      if (json["std_"][i].IsFloat()) {
        std_[i] = json["std_"][i].GetFloat();
      }
    }
  }
  return base::kStatusCodeOk;
}

TensorRtOnnxDetector::TensorRtOnnxDetector(const std::string &name)
    : dag::Node(name) {
  key_ = "nndeploy::detect::TensorRtOnnxDetector";
  desc_ =
      "ONNX detector with TensorRT backend [std::vector<cv::Mat>->DetectResult]";
  param_ = std::make_shared<TensorRtOnnxDetectorParam>();
  this->setInputTypeInfo<std::vector<cv::Mat>>("images");
  this->setOutputTypeInfo<DetectResult>("detection_result");
}

TensorRtOnnxDetector::TensorRtOnnxDetector(const std::string &name,
                                           std::vector<dag::Edge *> inputs,
                                           std::vector<dag::Edge *> outputs)
    : dag::Node(name, inputs, outputs) {
  key_ = "nndeploy::detect::TensorRtOnnxDetector";
  desc_ =
      "ONNX detector with TensorRT backend [std::vector<cv::Mat>->DetectResult]";
  param_ = std::make_shared<TensorRtOnnxDetectorParam>();
  if (inputs.empty()) {
    this->setInputTypeInfo<std::vector<cv::Mat>>("images");
  }
  if (outputs.empty()) {
    this->setOutputTypeInfo<DetectResult>("detection_result");
  }
}

TensorRtOnnxDetector::~TensorRtOnnxDetector() { this->deinit(); }

base::Status TensorRtOnnxDetector::init() {
  base::Status status = dag::Node::init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "node init failed");

  auto *param = dynamic_cast<TensorRtOnnxDetectorParam *>(param_.get());
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "param is nullptr");

  status = ensureInferenceReady(param);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "ensureInferenceReady failed");
  return status;
}

base::Status TensorRtOnnxDetector::deinit() {
  if (inference_) {
    inference_->deinit();
    inference_.reset();
  }
  inference_param_.reset();
  input_names_.clear();
  output_names_.clear();
  input_descs_.clear();
  host_device_ = nullptr;
  loaded_model_path_.clear();
  loaded_device_type_ = {base::kDeviceTypeCodeNone, 0};
  loaded_precision_ = base::kPrecisionTypeFp32;
  loaded_max_batch_size_ = 1;
  inference_ready_ = false;
  return dag::Node::deinit();
}

base::Status TensorRtOnnxDetector::run() {
  auto *param = dynamic_cast<TensorRtOnnxDetectorParam *>(param_.get());
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "param is nullptr");

  if (inputs_.empty()) {
    NNDEPLOY_LOGE("TensorRtOnnxDetector requires at least one input edge");
    return base::kStatusCodeErrorInvalidParam;
  }
  if (outputs_.empty()) {
    NNDEPLOY_LOGE("TensorRtOnnxDetector requires at least one output edge");
    return base::kStatusCodeErrorInvalidParam;
  }

  base::Status status = ensureInferenceReady(param);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "ensureInferenceReady failed");

  std::vector<cv::Mat> *images = inputs_[0]->get<std::vector<cv::Mat>>(this);
  if (images == nullptr) {
    NNDEPLOY_LOGE("Input images vector is nullptr");
    return base::kStatusCodeErrorInvalidParam;
  }

  if (images->empty()) {
    NNDEPLOY_LOGW("Input image list is empty");
  }

  if (input_names_.empty() || output_names_.empty()) {
    NNDEPLOY_LOGE("Inference tensors are not initialized");
    return base::kStatusCodeErrorInvalidState;
  }

  std::unique_ptr<DetectResult> aggregated = std::make_unique<DetectResult>();

  for (size_t image_index = 0; image_index < images->size(); ++image_index) {
    const cv::Mat &image = images->at(image_index);
    if (image.empty()) {
      NNDEPLOY_LOGW("Skip empty image at index %zu", image_index);
      continue;
    }

    std::vector<float> input_buffer;
    status = preprocessImage(image, param, input_buffer);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Preprocess failed for image %zu", image_index);
      continue;
    }

    const std::string &input_name = input_names_[0];
    const auto desc_iter = input_descs_.find(input_name);
    if (desc_iter == input_descs_.end()) {
      NNDEPLOY_LOGE("Unknown input tensor name: %s", input_name.c_str());
      return base::kStatusCodeErrorInvalidState;
    }

    device::TensorDesc desc = desc_iter->second;
    if (desc.shape_.size() < 4) {
      NNDEPLOY_LOGE("Unsupported input tensor shape size: %zu",
                    desc.shape_.size());
      return base::kStatusCodeErrorInvalidValue;
    }
    desc.shape_[0] = 1;
    desc.shape_[2] = param->input_height_;
    desc.shape_[3] = param->input_width_;

    if (host_device_ == nullptr) {
      host_device_ = device::getDefaultHostDevice();
    }
    device::Tensor host_tensor(host_device_, desc, input_name);
    float *tensor_data = reinterpret_cast<float *>(host_tensor.getData());
    size_t expected_size = input_buffer.size();
    size_t tensor_elements = 1;
    for (auto dim : desc.shape_) {
      tensor_elements *= static_cast<size_t>(dim);
    }
    if (tensor_elements != expected_size) {
      NNDEPLOY_LOGE("Input tensor size mismatch: %zu vs %zu", tensor_elements,
                    expected_size);
      return base::kStatusCodeErrorInvalidValue;
    }
    std::memcpy(tensor_data, input_buffer.data(),
                expected_size * sizeof(float));

    status = inference_->setInputTensor(input_name, &host_tensor);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "setInputTensor failed");

    status = inference_->run();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "inference run failed");

    device::Tensor *output_tensor = inference_->getOutputTensorAfterRun(
        output_names_[0], device::getDefaultHostDeviceType(), true);
    if (output_tensor == nullptr) {
      NNDEPLOY_LOGE("Failed to fetch output tensor");
      return base::kStatusCodeErrorNullParam;
    }

    DetectResult per_image;
    status =
        postprocessTensor(output_tensor, param, static_cast<int>(image_index),
                          per_image);
    delete output_tensor;
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Postprocess failed for image %zu", image_index);
      continue;
    }

    if (!per_image.bboxs_.empty()) {
      std::vector<int> keep(per_image.bboxs_.size());
      status = computeNMS(per_image, keep, param->nms_threshold_);
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("NMS failed for image %zu", image_index);
        continue;
      }
      for (int keep_index : keep) {
        if (keep_index < 0 || keep_index >= per_image.bboxs_.size()) {
          continue;
        }
        aggregated->bboxs_.push_back(per_image.bboxs_[keep_index]);
      }
    }
  }

  outputs_[0]->set(aggregated.release(), false);
  return base::kStatusCodeOk;
}

base::Status TensorRtOnnxDetector::ensureInferenceReady(
    TensorRtOnnxDetectorParam *param) {
  if (param->model_path_.empty()) {
    NNDEPLOY_LOGE("Model path is empty");
    return base::kStatusCodeErrorInvalidParam;
  }
  int max_batch = std::max(param->max_batch_size_, 1);
  if (inference_ready_ && param->model_path_ == loaded_model_path_ &&
      param->device_type_.code_ == loaded_device_type_.code_ &&
      param->device_type_.device_id_ == loaded_device_type_.device_id_ &&
      param->precision_type_ == loaded_precision_ &&
      max_batch == loaded_max_batch_size_) {
    return base::kStatusCodeOk;
  }

  std::ifstream file_stream(param->model_path_);
  if (!file_stream.good()) {
    NNDEPLOY_LOGE("Model file does not exist: %s", param->model_path_.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }

  if (param->input_width_ <= 0 || param->input_height_ <= 0) {
    NNDEPLOY_LOGE("Invalid input size: %dx%d", param->input_width_,
                  param->input_height_);
    return base::kStatusCodeErrorInvalidParam;
  }
  for (float value : param->std_) {
    if (std::abs(value) < 1e-6f) {
      NNDEPLOY_LOGE("Standard deviation must be non-zero");
      return base::kStatusCodeErrorInvalidParam;
    }
  }

  inference_ready_ = false;

  if (!inference_) {
    inference_ = inference::createInference(base::kInferenceTypeTensorRt);
    if (inference_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create TensorRT inference instance");
      return base::kStatusCodeErrorNullParam;
    }
  }

  if (!inference_param_) {
    inference_param_ =
        inference::createInferenceParam(base::kInferenceTypeTensorRt);
  }
  auto tensorrt_param = std::dynamic_pointer_cast<
      inference::TensorRtInferenceParam>(inference_param_);
  if (!tensorrt_param) {
    NNDEPLOY_LOGE("Failed to create TensorRT inference param");
    return base::kStatusCodeErrorNullParam;
  }

  inference_->deinit();

  tensorrt_param->device_type_ = param->device_type_;
  tensorrt_param->model_type_ = base::kModelTypeOnnx;
  tensorrt_param->model_value_.clear();
  tensorrt_param->model_value_.push_back(param->model_path_);
  tensorrt_param->is_path_ = true;
  tensorrt_param->max_batch_size_ = max_batch;
  tensorrt_param->precision_type_ = param->precision_type_;

  base::Status status = inference_->setParamSharedPtr(tensorrt_param);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "Failed to set inference param");

  if (stream_ != nullptr) {
    inference_->setStream(stream_);
  }
  status = inference_->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "TensorRT inference init failed");

  input_names_ = inference_->getAllInputTensorName();
  output_names_ = inference_->getAllOutputTensorName();
  input_descs_.clear();
  for (const auto &name : input_names_) {
    input_descs_[name] = inference_->getInputTensorDesc(name);
  }
  host_device_ = device::getDefaultHostDevice();
  loaded_model_path_ = param->model_path_;
  loaded_device_type_ = param->device_type_;
  loaded_precision_ = param->precision_type_;
  loaded_max_batch_size_ = max_batch;
  inference_ready_ = true;
  return base::kStatusCodeOk;
}

base::Status TensorRtOnnxDetector::preprocessImage(
    const cv::Mat &image, const TensorRtOnnxDetectorParam *param,
    std::vector<float> &buffer) const {
  if (image.empty()) {
    return base::kStatusCodeErrorInvalidParam;
  }
  cv::Mat converted;
  if (image.channels() == 1) {
    cv::cvtColor(image, converted, cv::COLOR_GRAY2BGR);
  } else {
    converted = image;
  }
  cv::Mat resized;
  cv::resize(converted, resized,
             cv::Size(param->input_width_, param->input_height_));
  if (param->swap_rb_) {
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
  }
  cv::Mat float_image;
  double scale = param->normalize_input_ ? 1.0 / 255.0 : 1.0;
  resized.convertTo(float_image, CV_32FC3, scale);

  std::vector<cv::Mat> channels(3);
  cv::split(float_image, channels);
  int spatial = param->input_width_ * param->input_height_;
  buffer.resize(static_cast<size_t>(spatial) * channels.size());
  for (int c = 0; c < static_cast<int>(channels.size()); ++c) {
    channels[c] = (channels[c] - param->mean_[c]) / param->std_[c];
    std::memcpy(buffer.data() + c * spatial, channels[c].ptr<float>(),
                spatial * sizeof(float));
  }
  return base::kStatusCodeOk;
}

base::Status TensorRtOnnxDetector::postprocessTensor(
    device::Tensor *tensor, const TensorRtOnnxDetectorParam *param,
    int image_index, DetectResult &result) const {
  if (tensor == nullptr) {
    return base::kStatusCodeErrorNullParam;
  }
  float *data = reinterpret_cast<float *>(tensor->getData());
  if (data == nullptr) {
    return base::kStatusCodeErrorNullParam;
  }

  base::IntVector shape = tensor->getShape();
  if (shape.empty()) {
    return base::kStatusCodeErrorInvalidValue;
  }

  int batch = 1;
  int num_boxes = 0;
  int attributes = 0;
  if (shape.size() == 2) {
    num_boxes = shape[0];
    attributes = shape[1];
  } else if (shape.size() == 3) {
    batch = shape[0];
    num_boxes = shape[1];
    attributes = shape[2];
  } else if (shape.size() == 4) {
    batch = shape[0];
    num_boxes = shape[2];
    attributes = shape[3];
  } else {
    NNDEPLOY_LOGE("Unsupported output tensor shape: %zu", shape.size());
    return base::kStatusCodeErrorInvalidValue;
  }

  if (attributes < 6) {
    NNDEPLOY_LOGE("Output tensor attribute size %d is insufficient", attributes);
    return base::kStatusCodeErrorInvalidValue;
  }

  if (image_index >= batch) {
    image_index = batch - 1;
  }
  size_t stride = static_cast<size_t>(num_boxes) * attributes;
  float *batch_ptr = data + static_cast<size_t>(image_index) * stride;

  for (int i = 0; i < num_boxes; ++i) {
    float *row = batch_ptr + static_cast<size_t>(i) * attributes;
    float x0 = 0.f;
    float y0 = 0.f;
    float x1 = 0.f;
    float y1 = 0.f;
    if (param->boxes_are_center_format_) {
      float x_center = row[0];
      float y_center = row[1];
      float width = row[2];
      float height = row[3];
      x0 = x_center - width * 0.5f;
      y0 = y_center - height * 0.5f;
      x1 = x_center + width * 0.5f;
      y1 = y_center + height * 0.5f;
    } else {
      x0 = row[0];
      y0 = row[1];
      x1 = row[2];
      y1 = row[3];
    }

    float best_score = row[4];
    int best_class = -1;
    if (param->has_class_probabilities_) {
      int class_count = attributes - 5;
      if (class_count <= 0) {
        continue;
      }
      best_score = 0.f;
      for (int c = 0; c < class_count; ++c) {
        float candidate = row[4] * row[5 + c];
        if (candidate > best_score) {
          best_score = candidate;
          best_class = c;
        }
      }
    } else {
      best_class = static_cast<int>(std::round(row[5]));
    }

    if (best_score < param->score_threshold_ || best_class < 0) {
      continue;
    }

    if (x1 < x0 || y1 < y0) {
      continue;
    }

    x0 = std::max(0.0f, std::min(x0, static_cast<float>(param->input_width_)));
    y0 = std::max(0.0f, std::min(y0, static_cast<float>(param->input_height_)));
    x1 = std::max(0.0f, std::min(x1, static_cast<float>(param->input_width_)));
    y1 = std::max(0.0f, std::min(y1, static_cast<float>(param->input_height_)));

    DetectBBoxResult bbox;
    bbox.index_ = image_index;
    bbox.label_id_ = best_class;
    bbox.score_ = best_score;
    bbox.bbox_[0] = x0 / static_cast<float>(param->input_width_);
    bbox.bbox_[1] = y0 / static_cast<float>(param->input_height_);
    bbox.bbox_[2] = x1 / static_cast<float>(param->input_width_);
    bbox.bbox_[3] = y1 / static_cast<float>(param->input_height_);
    result.bboxs_.push_back(bbox);
  }
  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::detect::TensorRtOnnxDetector", TensorRtOnnxDetector);

}  // namespace detect
}  // namespace nndeploy
