#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;

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

__device__ __forceinline__ float warp_sum(float value) {
    constexpr unsigned int kFullWarpMask = 0xffffffffu;
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullWarpMask, value, offset);
    }
    return value;
}

__global__ void residual_rmsnorm_fused_kernel(
    const at::Half* input,
    const at::Half* residual,
    const at::Half* weight,
    at::Half* output,
    int64_t hidden_size,
    float epsilon) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int64_t row_offset = row * hidden_size;
    const int thread = threadIdx.x;
    const int lane = thread & (kWarpSize - 1);
    const int warp = thread / kWarpSize;

    float partial_square_sum = 0.0f;
    for (int64_t column = thread; column < hidden_size; column += blockDim.x) {
        const int64_t index = row_offset + column;
        const float combined =
            static_cast<float>(input[index]) + static_cast<float>(residual[index]);
        partial_square_sum += combined * combined;
    }

    partial_square_sum = warp_sum(partial_square_sum);

    __shared__ float warp_sums[kWarpsPerBlock];
    if (lane == 0) {
        warp_sums[warp] = partial_square_sum;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum = lane < kWarpsPerBlock ? warp_sums[lane] : 0.0f;
        block_sum = warp_sum(block_sum);
        if (lane == 0) {
            warp_sums[0] = block_sum;
        }
    }
    __syncthreads();

    const float inverse_rms = rsqrtf(warp_sums[0] / hidden_size + epsilon);

    for (int64_t column = thread; column < hidden_size; column += blockDim.x) {
        const int64_t index = row_offset + column;
        const float combined =
            static_cast<float>(input[index]) + static_cast<float>(residual[index]);
        const float scaled =
            combined * inverse_rms * static_cast<float>(weight[column]);
        output[index] = static_cast<at::Half>(scaled);
    }
}

}  // namespace

torch::Tensor residual_rmsnorm_cuda(
    const torch::Tensor& input,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    double epsilon) {
    check_cuda_half_contiguous(input, "input");
    check_cuda_half_contiguous(residual, "residual");
    check_cuda_half_contiguous(weight, "weight");

    TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
    TORCH_CHECK(
        input.sizes() == residual.sizes(),
        "input and residual must have identical shapes");
    TORCH_CHECK(weight.dim() == 1, "weight must be one-dimensional");
    TORCH_CHECK(
        input.device() == residual.device() && input.device() == weight.device(),
        "input, residual, and weight must be on the same CUDA device");

    const int64_t hidden_size = input.size(-1);
    TORCH_CHECK(hidden_size > 0, "the final input dimension must be non-zero");
    TORCH_CHECK(input.numel() > 0, "input must contain at least one row");
    TORCH_CHECK(
        weight.size(0) == hidden_size,
        "weight length must equal the final input dimension");
    TORCH_CHECK(
        std::isfinite(epsilon) && epsilon > 0.0,
        "epsilon must be finite and greater than zero");

    const int64_t row_count = input.numel() / hidden_size;
    TORCH_CHECK(
        row_count <= static_cast<int64_t>(std::numeric_limits<unsigned int>::max()),
        "input has too many rows for the CUDA launch configuration");

    const c10::cuda::CUDAGuard device_guard(input.device());
    const cudaStream_t stream =
        c10::cuda::getCurrentCUDAStream(input.get_device()).stream();

    auto output = torch::empty_like(input);

    residual_rmsnorm_fused_kernel<<<
        static_cast<unsigned int>(row_count),
        kThreadsPerBlock,
        0,
        stream>>>(
        input.data_ptr<at::Half>(),
        residual.data_ptr<at::Half>(),
        weight.data_ptr<at::Half>(),
        output.data_ptr<at::Half>(),
        hidden_size,
        static_cast<float>(epsilon));

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
