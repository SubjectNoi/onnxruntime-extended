#include "contrib_ops/cpu/comm.h"
namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_TYPED_KERNEL_EX(
    Comm,
    kOnnxDomain,
    12,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Comm<float>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    Comm,
    kOnnxDomain,
    12,
    int64_t,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()),
    Comm<int64_t>
);


}
}