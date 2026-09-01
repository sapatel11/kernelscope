#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>

namespace {

constexpr int kThreadsPerBlock = 256;

void check_cuda_half_contiguous(
    const torch::Tensor& tensor,
    const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.scalar_type() == torch::kFloat16,
        name,
        " must have dtype float16");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

__global__ void swiglu_fused_kernel(
    const at::Half* gate,
    const at::Half* value,
    at::Half* output,
    int64_t element_count) {
    const int64_t index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= element_count) {
        return;
    }

    const float gate_value = static_cast<float>(gate[index]);
    const float value_value = static_cast<float>(value[index]);
    const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
    const float silu = gate_value * sigmoid;

    output[index] = static_cast<at::Half>(silu * value_value);
}

}  // namespace

torch::Tensor swiglu_cuda(
    const torch::Tensor& gate,
    const torch::Tensor& value) {
    check_cuda_half_contiguous(gate, "gate");
    check_cuda_half_contiguous(value, "value");
    TORCH_CHECK(
        gate.sizes() == value.sizes(),
        "gate and value must have identical shapes");
    TORCH_CHECK(
        gate.device() == value.device(),
        "gate and value must be on the same CUDA device");
    TORCH_CHECK(gate.numel() > 0, "gate and value must contain at least one element");

    const c10::cuda::CUDAGuard device_guard(gate.device());
    const cudaStream_t stream =
        c10::cuda::getCurrentCUDAStream(gate.get_device()).stream();

    auto output = torch::empty_like(gate);
    const int64_t element_count = gate.numel();
    const int64_t block_count =
        (element_count + kThreadsPerBlock - 1) / kThreadsPerBlock;

    swiglu_fused_kernel<<<
        static_cast<unsigned int>(block_count),
        kThreadsPerBlock,
        0,
        stream>>>(
        gate.data_ptr<at::Half>(),
        value.data_ptr<at::Half>(),
        output.data_ptr<at::Half>(),
        element_count);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
