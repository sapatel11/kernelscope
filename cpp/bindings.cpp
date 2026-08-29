#include <torch/extension.h>

torch::Tensor residual_rmsnorm_cpu(
    const torch::Tensor& input,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    double epsilon);

torch::Tensor swiglu_cpu(
    const torch::Tensor& gate,
    const torch::Tensor& value);

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "residual_rmsnorm_cpu",
        &residual_rmsnorm_cpu,
        py::arg("input"),
        py::arg("residual"),
        py::arg("weight"),
        py::arg("epsilon"),
        "Single-threaded CPU residual + RMSNorm reference");

    module.def(
        "swiglu_cpu",
        &swiglu_cpu,
        py::arg("gate"),
        py::arg("value"),
        "Single-threaded CPU SwiGLU reference");
}
