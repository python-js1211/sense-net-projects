#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("BigMLTreeify")
    .Input("points: float")
    .Input("split_indices: int32")
    .Input("split_values: float")
    .Input("left: int32")
    .Input("right: int32")
    .Input("outputs: float")
    .Output("output_features: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        auto npoints = c->Dim(c->input(0), 0);
        auto nprobs = c->Dim(c->input(5), 2);
        c->set_output(0, c->Matrix(npoints, nprobs));

        return Status::OK();
    });

class BigMLTreeOp : public OpKernel {
public:
    explicit BigMLTreeOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& points_tensor = context->input(0);
        const Tensor& sidx_tensor = context->input(1);
        const Tensor& sval_tensor = context->input(2);
        const Tensor& left_tensor = context->input(3);
        const Tensor& right_tensor = context->input(4);
        const Tensor& tree_outputs_tensor = context->input(5);

        const TensorShape& points_shape = points_tensor.shape();
        const TensorShape& tree_outputs_shape = tree_outputs_tensor.shape();

        auto points = points_tensor.matrix<float>();
        auto tree_outputs = tree_outputs_tensor.tensor<float, 3>();

        const int npts = points_shape.dim_size(0);
        const int ntrees = tree_outputs_shape.dim_size(0);
        const int nprobs = tree_outputs_shape.dim_size(2);

        // Create an output tensor
        Tensor* ot = NULL;
        const TensorShape out_shape = TensorShape({npts, nprobs});
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &ot));

        auto output_matrix = ot->matrix<float>();

        auto sidxs = sidx_tensor.matrix<int32>();
        auto svals = sval_tensor.matrix<float>();
        auto left = left_tensor.matrix<int32>();
        auto right = right_tensor.matrix<int32>();

        int current_idx, split_index, ti, pi, oi, leaf;
        float output;

        for (pi = 0; pi < npts; pi++) {
            for (oi = 0; oi < nprobs; oi++) {
                output_matrix(pi, oi) = 0;
            }
        }

        for (ti = 0; ti < ntrees; ti++) {
            for (pi = 0; pi < npts; pi++) {
                current_idx = 0;
                split_index = sidxs(ti, 0);

                while (split_index >= 0) {
                    if (points(pi, split_index) <= svals(ti, current_idx))
                        current_idx = left(ti, current_idx);
                    else
                        current_idx = right(ti, current_idx);

                    split_index = sidxs(ti, current_idx);
                }

                leaf = left(ti, current_idx);

                for (oi = 0; oi < nprobs; oi++) {
                    output = output_matrix(pi, oi);
                    output_matrix(pi, oi) = output + tree_outputs(ti, leaf, oi);
                }
            }
        }

        for (pi = 0; pi < npts; pi++) {
            for (oi = 0; oi < nprobs; oi++) {
                output_matrix(pi, oi) = output_matrix(pi, oi) / ntrees;
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("BigMLTreeify").Device(DEVICE_CPU), BigMLTreeOp);
