#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"
#include <unsupported/Eigen/SpecialFunctions>
#include "core/providers/cpu/element_wise_ranged_transform.h"

namespace onnxruntime {
namespace contrib {
template <typename T>
class Comm : public OpKernel {
public:
    Comm(const OpKernelInfo& info) : OpKernel(info) {
        std::string _sub_type;
        if (info.GetAttr<std::string>("sub_type", &_sub_type).IsOK()) {
            sub_type = _sub_type;
        }
        int64_t _tag;
        if (info.GetAttr<int64_t>("comm_tag", &_tag).IsOK()) {
            tag = _tag;
        }
        std::string _target;
        if (info.GetAttr<std::string>("comm_target", &_target).IsOK()) {
            target = _target;
        }
    }

    Status Compute(OpKernelContext* context) const override {
        const Tensor* input = context->Input<Tensor>(0);
        const T* input_data = input->template Data<T>();

        Tensor* output = context->Output(0, input->Shape());
        T* output_data = output->template MutableData<T>();

        auto in_shape = input->Shape();
        std::cout << "Computing Comm Node " << tag << ", Target " << target << ", Subtype " << sub_type << ", Input Size Dim " << in_shape.NumDimensions() << "\n";
        for (size_t i = 0; i < in_shape.NumDimensions(); i++) {
            std::cout << in_shape[i] << " ";
        }
        std::cout << "\n";
        for (int i = 0; i < input->Shape().Size(); i++) {
            output_data[i] = input_data[i];
        }
        return Status::OK();
    }
private:
    std::string sub_type;
    std::string target;
    int64_t tag;
};
}
}