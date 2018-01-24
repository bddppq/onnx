#include "onnx/backend/registry.h"

namespace onnx {
namespace backend {
Registry::Map& Registry::map() {
  static Map map;
  return map;
}
} // namespace backend
} // namespace onnx
