#pragma once

#include <string>
#include <unordered_map>
#include "onnx/backend/base.h"

namespace onnx {
namespace backend {

class Registry {
 public:
  using Map = std::unordered_map<std::string, Backend*>;
  Map& map();
};
} // namespace backend
} // namespace onnx
