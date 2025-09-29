#include <algorithm>
#include <array>
#include <cctype>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "flag.h"
#include "nndeploy/base/file.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/status.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/detect/tensorrt_onnx_detector.h"
#include "nndeploy/framework.h"

using namespace nndeploy;

DEFINE_string(tensorrt_detector_model_path, "",
              "Path to the ONNX model that TensorRT will consume.");
DEFINE_string(tensorrt_detector_image_paths, "",
              "Comma separated list of images that should be processed.");
DEFINE_int32(tensorrt_detector_input_width, 640,
             "Network input width for preprocessing.");
DEFINE_int32(tensorrt_detector_input_height, 640,
             "Network input height for preprocessing.");
DEFINE_double(tensorrt_detector_score_threshold, 0.5,
              "Minimum confidence required to keep detections.");
DEFINE_double(tensorrt_detector_nms_threshold, 0.45,
              "IoU threshold used during non-maximum suppression.");
DEFINE_int32(tensorrt_detector_num_classes, 80,
             "Number of categories predicted by the model output.");
DEFINE_int32(tensorrt_detector_max_batch_size, 1,
             "Maximum batch size to use when building the TensorRT engine.");
DEFINE_bool(tensorrt_detector_normalize_input, true,
            "Whether to scale pixel values to [0,1] during preprocessing.");
DEFINE_bool(tensorrt_detector_swap_rb, true,
            "Swap red and blue channels during preprocessing.");
DEFINE_bool(tensorrt_detector_boxes_are_center_format, true,
            "Indicates if model boxes are encoded as center x/y and width/height.");
DEFINE_bool(tensorrt_detector_has_class_probabilities, true,
            "Expect per-class probabilities in the output tensor.");
DEFINE_string(tensorrt_detector_mean, "0.0,0.0,0.0",
              "Comma separated per-channel mean used during preprocessing.");
DEFINE_string(tensorrt_detector_std, "1.0,1.0,1.0",
              "Comma separated per-channel std used during preprocessing.");

namespace {

std::vector<std::string> splitCommaSeparated(const std::string &value) {
  std::vector<std::string> tokens;
  std::string token;
  std::stringstream ss(value);
  while (std::getline(ss, token, ',')) {
    if (!token.empty()) {
      // Remove surrounding whitespace
      token.erase(token.begin(),
                  std::find_if(token.begin(), token.end(), [](unsigned char c) {
                    return !std::isspace(c);
                  }));
      token.erase(std::find_if(token.rbegin(), token.rend(), [](unsigned char c) {
                    return !std::isspace(c);
                  }).base(),
                  token.end());
      if (!token.empty()) {
        tokens.push_back(token);
      }
    }
  }
  return tokens;
}

std::array<float, 3> parseFloatArray3(const std::string &value,
                                      const std::array<float, 3> &fallback) {
  std::array<float, 3> parsed = fallback;
  auto tokens = splitCommaSeparated(value);
  for (size_t i = 0; i < std::min<size_t>(tokens.size(), parsed.size()); ++i) {
    try {
      parsed[i] = std::stof(tokens[i]);
    } catch (const std::exception &) {
      NNDEPLOY_LOGW("Failed to parse value '%s' as float, using fallback %.3f",
                    tokens[i].c_str(), parsed[i]);
    }
  }
  return parsed;
}

std::vector<std::string> resolveImagePaths() {
  std::vector<std::string> resolved;
  auto from_flag = splitCommaSeparated(FLAGS_tensorrt_detector_image_paths);
  resolved.insert(resolved.end(), from_flag.begin(), from_flag.end());

  if (resolved.empty()) {
    std::string fallback = demo::getInputPath();
    if (!fallback.empty()) {
      if (base::isDirectory(fallback)) {
        auto files = demo::getAllFileFromDir(fallback);
        resolved.insert(resolved.end(), files.begin(), files.end());
        std::sort(resolved.begin(), resolved.end());
      } else {
        resolved.push_back(fallback);
      }
    }
  }

  resolved.erase(std::remove_if(resolved.begin(), resolved.end(),
                                [](const std::string &path) {
                                  return path.empty();
                                }),
                 resolved.end());
  return resolved;
}

void printUsage() {
  std::cout << "TensorRT ONNX detector demo" << std::endl;
  std::cout << "Required flags:" << std::endl;
  std::cout << "  --tensorrt_detector_model_path=/path/to/model.onnx" << std::endl;
  std::cout << "Optional flags:" << std::endl;
  std::cout << "  --tensorrt_detector_image_paths=/path/a.jpg,/path/b.jpg"
            << std::endl;
  std::cout << "  --input_path can be used as a fallback when image paths are not"
               " provided"
            << std::endl;
  std::cout << "  --device_type=kDeviceTypeCodeCuda:0 (defaults to CUDA device 0)"
            << std::endl;
  std::cout << "  --precision_type=kPrecisionTypeFp16 (optional)" << std::endl;
  std::cout << std::endl;
}

}  // namespace

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (demo::FLAGS_usage) {
    demo::showUsage();
    printUsage();
    return -1;
  }

  if (FLAGS_tensorrt_detector_model_path.empty()) {
    printUsage();
    NNDEPLOY_LOGE("--tensorrt_detector_model_path must be specified");
    return -1;
  }

  int ret = nndeployFrameworkInit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d", ret);
    return ret;
  }

  dag::Edge image_edge("detector_input_images");
  dag::Edge detection_edge("detector_output");
  dag::Graph detector_graph("TensorRtOnnxDetectorGraph", {&image_edge},
                            {&detection_edge});

  auto *detector = new detect::TensorRtOnnxDetector(
      "TensorRtOnnxDetector", std::vector<dag::Edge *>{&image_edge},
      std::vector<dag::Edge *>{&detection_edge});
  detector_graph.addNode(detector, false);

  auto *param = dynamic_cast<detect::TensorRtOnnxDetectorParam *>(
      detector->getParam());
  if (param == nullptr) {
    NNDEPLOY_LOGE("Failed to access detector parameters");
    return -1;
  }

  param->model_path_ = FLAGS_tensorrt_detector_model_path;
  param->input_width_ = FLAGS_tensorrt_detector_input_width;
  param->input_height_ = FLAGS_tensorrt_detector_input_height;
  param->score_threshold_ = static_cast<float>(
      std::max(0.0, std::min(1.0, FLAGS_tensorrt_detector_score_threshold)));
  param->nms_threshold_ = static_cast<float>(
      std::max(0.0, std::min(1.0, FLAGS_tensorrt_detector_nms_threshold)));
  param->num_classes_ = FLAGS_tensorrt_detector_num_classes;
  param->max_batch_size_ =
      std::max(1, FLAGS_tensorrt_detector_max_batch_size);
  param->normalize_input_ = FLAGS_tensorrt_detector_normalize_input;
  param->swap_rb_ = FLAGS_tensorrt_detector_swap_rb;
  param->boxes_are_center_format_ =
      FLAGS_tensorrt_detector_boxes_are_center_format;
  param->has_class_probabilities_ =
      FLAGS_tensorrt_detector_has_class_probabilities;
  param->mean_ =
      parseFloatArray3(FLAGS_tensorrt_detector_mean, param->mean_);
  param->std_ = parseFloatArray3(FLAGS_tensorrt_detector_std, param->std_);

  if (!demo::FLAGS_device_type.empty()) {
    param->device_type_ = demo::getDeviceType();
  }
  if (!demo::FLAGS_precision_type.empty()) {
    param->precision_type_ = demo::getPrecisionType();
  }

  detector_graph.markOutputEdge({&detection_edge});

  base::Status status = detector_graph.init();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Failed to initialize detector graph: %s",
                  base::statusCodeToString(status).c_str());
    nndeployFrameworkDeinit();
    return -1;
  }

  std::vector<std::string> image_paths = resolveImagePaths();
  if (image_paths.empty()) {
    NNDEPLOY_LOGE("No input images were provided");
    detector_graph.deinit();
    nndeployFrameworkDeinit();
    return -1;
  }

  std::vector<cv::Mat> images;
  images.reserve(image_paths.size());
  std::vector<std::string> loaded_paths;
  loaded_paths.reserve(image_paths.size());
  for (const auto &path : image_paths) {
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
      NNDEPLOY_LOGW("Failed to load image: %s", path.c_str());
      continue;
    }
    images.emplace_back(image);
    loaded_paths.emplace_back(path);
  }

  if (images.empty()) {
    NNDEPLOY_LOGE("All provided images failed to load");
    detector_graph.deinit();
    nndeployFrameworkDeinit();
    return -1;
  }

  param->max_batch_size_ =
      std::max(param->max_batch_size_, static_cast<int>(images.size()));

  image_edge.set(&images, true);

  status = detector_graph.run();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Detector graph execution failed: %s",
                  base::statusCodeToString(status).c_str());
    detector_graph.deinit();
    nndeployFrameworkDeinit();
    return -1;
  }

  auto *result =
      detection_edge.getGraphOutput<detect::DetectResult>();
  if (result == nullptr) {
    NNDEPLOY_LOGE("Detector result is null");
    detector_graph.deinit();
    nndeployFrameworkDeinit();
    return -1;
  }

  std::cout << "Detections collected from TensorRtOnnxDetector:" << std::endl;
  for (size_t img_index = 0; img_index < images.size(); ++img_index) {
    std::cout << "Image " << img_index << " (" << loaded_paths[img_index]
              << ", " << images[img_index].cols << "x" << images[img_index].rows
              << ")" << std::endl;
    bool printed_any = false;
    for (const auto &bbox : result->bboxs_) {
      if (bbox.index_ != static_cast<int>(img_index)) {
        continue;
      }
      printed_any = true;
      int x0 = static_cast<int>(bbox.bbox_[0] * images[img_index].cols);
      int y0 = static_cast<int>(bbox.bbox_[1] * images[img_index].rows);
      int x1 = static_cast<int>(bbox.bbox_[2] * images[img_index].cols);
      int y1 = static_cast<int>(bbox.bbox_[3] * images[img_index].rows);
      std::cout << "  class=" << bbox.label_id_ << " score=" << bbox.score_
                << " box=[" << x0 << ", " << y0 << ", " << x1 << ", " << y1
                << "]" << std::endl;
    }
    if (!printed_any) {
      std::cout << "  (no detections above threshold)" << std::endl;
    }
  }

  detector_graph.deinit();

  ret = nndeployFrameworkDeinit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkDeinit failed. ERROR: %d", ret);
    return ret;
  }

  return 0;
}
