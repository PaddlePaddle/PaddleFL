#include "./paddle_tensor.h"
#include "./paddle_tensor_impl.cu.h"

namespace aby3 {
template class PaddleTensor<int64_t>;
} // namespace aby3
