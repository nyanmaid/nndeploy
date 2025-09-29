#ifndef _NNDEPLOY_DETECT_TENSORRT_ONNX_DETECTOR_H_
#define _NNDEPLOY_DETECT_TENSORRT_ONNX_DETECTOR_H_

#include <array>
#include <map>
#include <string>
#include <vector>

#include "nndeploy/base/param.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/status.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/detect/result.h"
#include "nndeploy/device/type.h"
#include "nndeploy/inference/inference.h"

namespace nndeploy {
namespace detect {

class NNDEPLOY_CC_API TensorRtOnnxDetectorParam : public base::Param {
 public:
  std::string model_path_ = "";
  int input_width_ = 640;
  int input_height_ = 640;
  float score_threshold_ = 0.5f;
  float nms_threshold_ = 0.45f;
  int num_classes_ = 80;
  bool normalize_input_ = true;
  bool swap_rb_ = true;
  bool boxes_are_center_format_ = true;
  bool has_class_probabilities_ = true;
  std::array<float, 3> mean_ = {0.0f, 0.0f, 0.0f};
  std::array<float, 3> std_ = {1.0f, 1.0f, 1.0f};
  int max_batch_size_ = 1;
  base::PrecisionType precision_type_ = base::kPrecisionTypeFp32;
  base::DeviceType device_type_ = {base::kDeviceTypeCodeCuda, 0};

  using base::Param::serialize;
  base::Status serialize(rapidjson::Value &json,
                         rapidjson::Document::AllocatorType &allocator) override;
  using base::Param::deserialize;
  base::Status deserialize(rapidjson::Value &json) override;
};

class NNDEPLOY_CC_API TensorRtOnnxDetector : public dag::Node {
 public:
  explicit TensorRtOnnxDetector(const std::string &name);
  TensorRtOnnxDetector(const std::string &name, std::vector<dag::Edge *> inputs,
                       std::vector<dag::Edge *> outputs);
  ~TensorRtOnnxDetector() override;

  base::Status init() override;
  base::Status deinit() override;
  base::Status run() override;

 private:
  base::Status ensureInferenceReady(TensorRtOnnxDetectorParam *param);
  base::Status preprocessImage(const cv::Mat &image,
                               const TensorRtOnnxDetectorParam *param,
                               std::vector<float> &buffer) const;
  base::Status postprocessTensor(device::Tensor *tensor,
                                 const TensorRtOnnxDetectorParam *param,
                                 int image_index, DetectResult &result) const;

  std::shared_ptr<inference::Inference> inference_;
  std::shared_ptr<inference::InferenceParam> inference_param_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::map<std::string, device::TensorDesc> input_descs_;
  device::Device *host_device_ = nullptr;
  std::string loaded_model_path_;
  base::DeviceType loaded_device_type_ = {base::kDeviceTypeCodeNone, 0};
  base::PrecisionType loaded_precision_ = base::kPrecisionTypeFp32;
  int loaded_max_batch_size_ = 1;
  bool inference_ready_ = false;
};

}  // namespace detect
}  // namespace nndeploy

#endif  // _NNDEPLOY_DETECT_TENSORRT_ONNX_DETECTOR_H_
