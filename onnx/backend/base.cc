#include "onnx/backend/base.h"

namespace onnx {
namespace backend {

bool Backend::supports_device(const Device& device) {
  return true;
}

bool Backend::supports_model(const ModelProto& model, const Device& device) {
  return supports_graph(model.graph());
}

bool Backend::supports_graph(const GraphProto& graph, const Device& device) {
  for (auto const& node : graph.node()) {
    if (!supports_node(node, device)) {
      return false;
    }
  }
  return true;
}

BackendRep* Backend::prepare(const ModelProto& model, const Device& device) {
  checker::check_model(model);
  return prepare(model.graph(), device);
}

bool Backend::run_model(
    const ModelProto& model,
    const DLManagedTensor* inputs,
    DLManagedTensor* outputs,
    const Device& device) {
  checker::check_model(model);
  return run_graph(model.graph(), inputs, outputs, device);
}

} // namespace backend
} // namespace onnx
