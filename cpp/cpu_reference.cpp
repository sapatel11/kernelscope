#include <torch/extension.h>

#include <cmath>
#include <cstdint>

namespace {

void check_cpu_float32_contiguous(
    const torch::Tensor& tensor,
    const char* name) {
    TORCH_CHECK(!tensor.is_cuda(), name, " must be a CPU tensor");
    TORCH_CHECK(
        tensor.scalar_type() == torch::kFloat32,
        name,
        " must have dtype float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor residual_rmsnorm_cpu(
    const torch::Tensor& input,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    double epsilon) {
    check_cpu_float32_contiguous(input, "input");
    check_cpu_float32_contiguous(residual, "residual");
    check_cpu_float32_contiguous(weight, "weight");

    TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
    TORCH_CHECK(
        input.sizes() == residual.sizes(),
        "input and residual must have identical shapes");
    TORCH_CHECK(weight.dim() == 1, "weight must be one-dimensional");

    const int64_t hidden_size = input.size(-1);
    TORCH_CHECK(hidden_size > 0, "the final input dimension must be non-zero");
    TORCH_CHECK(
        weight.size(0) == hidden_size,
        "weight length must equal the final input dimension");
    TORCH_CHECK(
        std::isfinite(epsilon) && epsilon > 0.0,
        "epsilon must be finite and greater than zero");

    auto combined = torch::empty_like(input);
    auto output = torch::empty_like(input);

    const float* input_data = input.data_ptr<float>();
    const float* residual_data = residual.data_ptr<float>();
    const float* weight_data = weight.data_ptr<float>();
    float* combined_data = combined.data_ptr<float>();
    float* output_data = output.data_ptr<float>();

    const int64_t row_count = input.numel() / hidden_size;
    for (int64_t row = 0; row < row_count; ++row) {
        const int64_t row_offset = row * hidden_size;
        double square_sum = 0.0;

        for (int64_t column = 0; column < hidden_size; ++column) {
            const int64_t index = row_offset + column;
            const float combined_value = input_data[index] + residual_data[index];
            combined_data[index] = combined_value;
            square_sum +=
                static_cast<double>(combined_value) * combined_value;
        }

        const double mean_square = square_sum / static_cast<double>(hidden_size);
        const float inverse_rms =
            static_cast<float>(1.0 / std::sqrt(mean_square + epsilon));

        for (int64_t column = 0; column < hidden_size; ++column) {
            const int64_t index = row_offset + column;
            output_data[index] =
                combined_data[index] * inverse_rms * weight_data[column];
        }
    }

    return output;
}

torch::Tensor swiglu_cpu(
    const torch::Tensor& gate,
    const torch::Tensor& value) {
    check_cpu_float32_contiguous(gate, "gate");
    check_cpu_float32_contiguous(value, "value");
    TORCH_CHECK(
        gate.sizes() == value.sizes(),
        "gate and value must have identical shapes");

    auto output = torch::empty_like(gate);
    const float* gate_data = gate.data_ptr<float>();
    const float* value_data = value.data_ptr<float>();
    float* output_data = output.data_ptr<float>();

    for (int64_t index = 0; index < gate.numel(); ++index) {
        const double gate_value = gate_data[index];
        const double silu_value =
            gate_value / (1.0 + std::exp(-gate_value));
        output_data[index] =
            static_cast<float>(silu_value * value_data[index]);
    }

    return output;
}
