#pragma once

#include <dlpack/dlpack.h>
#include <stdexcept>
#include "onnx/checker.h"
#include "onnx/onnx_pb.h"

namespace onnx {
namespace backend {

struct BackendError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct ExecutionError : BackendError {
  using BackendError::BackendError;
};

struct NotImplemented : BackendError {
  using BackendError::BackendError;
};

using DeviceType = DLDeviceType;
using Device = DLContext;

struct BackendRep {
  virtual bool run(const DLManagedTensor* inputs, DLManagedTensor* outputs) = 0;
  virtual ~BackendRep() noexcept;
};

struct Backend {
  virtual bool supports_device(const Device& device);
  virtual bool supports_model(
      const ModelProto& model,
      const Device& device = {.device_type = DLDeviceType::kDLCPU});
  virtual bool supports_graph(
      const GraphProto& graph,
      const Device& device = {.device_type = DLDeviceType::kDLCPU});
  virtual bool supports_node(
      const NodeProto& node,
      const Device& device = {.device_type = DLDeviceType::kDLCPU}) = 0;

  virtual BackendRep* prepare(
      const ModelProto& model,
      const Device& device = {.device_type = DLDeviceType::kDLCPU});
  virtual BackendRep* prepare(
      const GraphProto& graph,
      const Device& device = {.device_type = DLDeviceType::kDLCPU}) = 0;
  virtual bool run_model(
      const ModelProto& model,
      const DLManagedTensor* inputs,
      DLManagedTensor* outputs,
      const Device& device = {.device_type = DLDeviceType::kDLCPU});
  virtual bool run_graph(
      const GraphProto& graph,
      const DLManagedTensor* inputs,
      DLManagedTensor* outputs,
      const Device& device = {.device_type = DLDeviceType::kDLCPU}) = 0;
  virtual bool run_node(
      const NodeProto& node,
      const DLManagedTensor* inputs,
      DLManagedTensor* outputs,
      const Device& device = {.device_type = DLDeviceType::kDLCPU}) = 0;

  virtual ~Backend() noexcept;
};

} // namespace backend
} // namespace onnx
