#include "./paddle_tensor.h"
#include "./paddle_tensor_impl.cu.h"

namespace common {
template
void MatMul<int64_t>::mat_mul(const TensorAdapter<int64_t>* lhs,
                              const TensorAdapter<int64_t>* rhs,
                              TensorAdapter<int64_t>* ret,
                              bool trans_lhs,
                              bool trans_rhs);
} // namespace common
