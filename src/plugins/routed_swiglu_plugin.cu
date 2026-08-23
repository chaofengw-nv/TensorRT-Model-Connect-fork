/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#if TRTMC_HAS_TRT

#include "plugins/routed_swiglu_plugin.h"

#include <NvInferPlugin.h>
#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace trtmc {
namespace {

constexpr int32_t kThreads = 256;
constexpr int32_t kDecodeSplitK = 8;

__global__ void routed_gate_up_kernel(__half const* input, int32_t const* expert_indices,
                                      __half const* gate_weights, __half const* up_weights,
                                      __half* gated, int32_t rows, int32_t top_k,
                                      int32_t num_experts, int32_t hidden_size,
                                      int32_t intermediate_size) {
    const int32_t column = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int32_t row = static_cast<int32_t>(blockIdx.y);
    const int32_t route = static_cast<int32_t>(blockIdx.z);
    if (row >= rows || route >= top_k || column >= intermediate_size)
        return;

    const int32_t expert = expert_indices[row * top_k + route];
    if (expert < 0 || expert >= num_experts)
        return;

    float gate = 0.0F;
    float up = 0.0F;
    const size_t expert_offset = static_cast<size_t>(expert) * hidden_size * intermediate_size;
    for (int32_t hidden = 0; hidden < hidden_size; ++hidden) {
        const float activation = __half2float(input[row * hidden_size + hidden]);
        const size_t weight_offset =
            expert_offset + static_cast<size_t>(hidden) * intermediate_size + column;
        gate = fmaf(activation, __half2float(gate_weights[weight_offset]), gate);
        up = fmaf(activation, __half2float(up_weights[weight_offset]), up);
    }
    const float swish = gate / (1.0F + expf(-gate));
    gated[(row * top_k + route) * intermediate_size + column] = __float2half(swish * up);
}

__global__ void routed_gate_up_split_kernel(__half const* input, int32_t const* expert_indices,
                                            __half const* gate_weights, __half const* up_weights,
                                            float* gate_partials, float* up_partials, int32_t top_k,
                                            int32_t num_experts, int32_t hidden_size,
                                            int32_t intermediate_size) {
    const int32_t column = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int32_t route_and_split = static_cast<int32_t>(blockIdx.y);
    const int32_t route = route_and_split / kDecodeSplitK;
    const int32_t split = route_and_split % kDecodeSplitK;
    if (route >= top_k || column >= intermediate_size)
        return;

    const int32_t expert = expert_indices[route];
    if (expert < 0 || expert >= num_experts)
        return;
    float gate = 0.0F;
    float up = 0.0F;
    const size_t expert_offset = static_cast<size_t>(expert) * hidden_size * intermediate_size;
    for (int32_t hidden = split; hidden < hidden_size; hidden += kDecodeSplitK) {
        const float activation = __half2float(input[hidden]);
        const size_t weight_offset =
            expert_offset + static_cast<size_t>(hidden) * intermediate_size + column;
        gate = fmaf(activation, __half2float(gate_weights[weight_offset]), gate);
        up = fmaf(activation, __half2float(up_weights[weight_offset]), up);
    }
    const size_t partial_offset =
        (static_cast<size_t>(route_and_split) * intermediate_size) + column;
    gate_partials[partial_offset] = gate;
    up_partials[partial_offset] = up;
}

__global__ void routed_gate_up_reduce_kernel(float const* gate_partials, float const* up_partials,
                                             __half* gated, int32_t top_k,
                                             int32_t intermediate_size) {
    const int32_t column = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int32_t route = static_cast<int32_t>(blockIdx.y);
    if (route >= top_k || column >= intermediate_size)
        return;
    float gate = 0.0F;
    float up = 0.0F;
    for (int32_t split = 0; split < kDecodeSplitK; ++split) {
        const size_t partial_offset =
            (static_cast<size_t>(route * kDecodeSplitK + split) * intermediate_size) + column;
        gate += gate_partials[partial_offset];
        up += up_partials[partial_offset];
    }
    const float swish = gate / (1.0F + expf(-gate));
    gated[route * intermediate_size + column] = __float2half(swish * up);
}

__global__ void routed_down_kernel(__half const* gated, int32_t const* expert_indices,
                                   __half const* route_weights, __half const* down_weights,
                                   __half* output, int32_t rows, int32_t top_k, int32_t num_experts,
                                   int32_t hidden_size, int32_t intermediate_size) {
    const int32_t column = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int32_t row = static_cast<int32_t>(blockIdx.y);
    if (row >= rows || column >= hidden_size)
        return;

    float combined = 0.0F;
    for (int32_t route = 0; route < top_k; ++route) {
        const int32_t expert = expert_indices[row * top_k + route];
        if (expert < 0 || expert >= num_experts)
            continue;
        float projected = 0.0F;
        const size_t expert_offset = static_cast<size_t>(expert) * intermediate_size * hidden_size;
        const __half* route_input = gated + (row * top_k + route) * intermediate_size;
        for (int32_t intermediate = 0; intermediate < intermediate_size; ++intermediate) {
            const size_t weight_offset =
                expert_offset + static_cast<size_t>(intermediate) * hidden_size + column;
            projected = fmaf(__half2float(route_input[intermediate]),
                             __half2float(down_weights[weight_offset]), projected);
        }
        combined = fmaf(__half2float(route_weights[row * top_k + route]), projected, combined);
    }
    output[row * hidden_size + column] = __float2half(combined);
}

} // namespace

class RoutedSwiGluCreator final : public nvinfer1::IPluginCreator {
  public:
    RoutedSwiGluCreator() {
        fields_storage_.emplace_back(
            nvinfer1::PluginField{"max_rows", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.nbFields = static_cast<int32_t>(fields_storage_.size());
        fields_.fields = fields_storage_.data();
    }

    char const* getPluginName() const noexcept override { return RoutedSwiGluPlugin::kPLUGIN_NAME; }
    char const* getPluginVersion() const noexcept override {
        return RoutedSwiGluPlugin::kPLUGIN_VERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    void setPluginNamespace(char const* plugin_namespace) noexcept override {
        namespace_ = plugin_namespace == nullptr ? "" : plugin_namespace;
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

    nvinfer1::IPluginV2*
    createPlugin(char const*, nvinfer1::PluginFieldCollection const* fields) noexcept override {
        try {
            int32_t max_rows = 0;
            if (fields != nullptr) {
                for (int32_t index = 0; index < fields->nbFields; ++index) {
                    const auto& field = fields->fields[index];
                    if (field.name != nullptr && std::strcmp(field.name, "max_rows") == 0 &&
                        field.data != nullptr && field.length == 1) {
                        max_rows = *static_cast<int32_t const*>(field.data);
                    }
                }
            }
            if (max_rows <= 0)
                return nullptr;
            auto plugin = std::make_unique<RoutedSwiGluPlugin>(max_rows);
            plugin->setPluginNamespace(namespace_.c_str());
            return plugin.release();
        } catch (...) {
            return nullptr;
        }
    }

    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        if (data == nullptr || length != sizeof(int32_t))
            return nullptr;
        int32_t max_rows = 0;
        std::memcpy(&max_rows, data, sizeof(max_rows));
        if (max_rows <= 0)
            return nullptr;
        try {
            auto plugin = std::make_unique<RoutedSwiGluPlugin>(max_rows);
            plugin->setPluginNamespace(namespace_.c_str());
            return plugin.release();
        } catch (...) {
            return nullptr;
        }
    }

  private:
    std::vector<nvinfer1::PluginField> fields_storage_;
    nvinfer1::PluginFieldCollection fields_{};
    std::string namespace_;
};

char const* RoutedSwiGluPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}
char const* RoutedSwiGluPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}
int32_t RoutedSwiGluPlugin::getNbOutputs() const noexcept {
    return 1;
}
int32_t RoutedSwiGluPlugin::initialize() noexcept {
    return 0;
}
void RoutedSwiGluPlugin::terminate() noexcept {}
void RoutedSwiGluPlugin::destroy() noexcept {
    delete this;
}
size_t RoutedSwiGluPlugin::getSerializationSize() const noexcept {
    return sizeof(max_rows_);
}
void RoutedSwiGluPlugin::serialize(void* buffer) const noexcept {
    std::memcpy(buffer, &max_rows_, sizeof(max_rows_));
}
void RoutedSwiGluPlugin::setPluginNamespace(char const* plugin_namespace) noexcept {
    namespace_ = plugin_namespace == nullptr ? "" : plugin_namespace;
}
char const* RoutedSwiGluPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType RoutedSwiGluPlugin::getOutputDataType(int32_t,
                                                         nvinfer1::DataType const* input_types,
                                                         int32_t) const noexcept {
    return input_types[0];
}

RoutedSwiGluPlugin* RoutedSwiGluPlugin::clone() const noexcept {
    try {
        auto plugin = std::make_unique<RoutedSwiGluPlugin>(max_rows_);
        plugin->namespace_ = namespace_;
        return plugin.release();
    } catch (...) {
        return nullptr;
    }
}

nvinfer1::DimsExprs RoutedSwiGluPlugin::getOutputDimensions(int32_t,
                                                            nvinfer1::DimsExprs const* inputs,
                                                            int32_t,
                                                            nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}

bool RoutedSwiGluPlugin::supportsFormatCombination(int32_t position,
                                                   nvinfer1::PluginTensorDesc const* input_output,
                                                   int32_t input_count,
                                                   int32_t output_count) noexcept {
    if (input_count != 6 || output_count != 1 || position < 0 || position >= 7)
        return false;
    if (input_output[position].format != nvinfer1::TensorFormat::kLINEAR)
        return false;
    if (position == 1)
        return input_output[position].type == nvinfer1::DataType::kINT32;
    return input_output[position].type == nvinfer1::DataType::kHALF;
}

void RoutedSwiGluPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                         nvinfer1::DynamicPluginTensorDesc const*,
                                         int32_t) noexcept {}

size_t RoutedSwiGluPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs,
                                            int32_t input_count, nvinfer1::PluginTensorDesc const*,
                                            int32_t) const noexcept {
    if (inputs == nullptr || input_count != 6 || inputs[1].dims.nbDims != 2 ||
        inputs[3].dims.nbDims != 3)
        return 0;
    const int32_t top_k = inputs[1].dims.d[1];
    const int32_t intermediate_size = inputs[3].dims.d[2];
    if (top_k <= 0 || intermediate_size <= 0)
        return 0;
    const size_t gated_bytes =
        static_cast<size_t>(max_rows_) * top_k * intermediate_size * sizeof(__half);
    const size_t decode_partials_bytes =
        static_cast<size_t>(2) * top_k * kDecodeSplitK * intermediate_size * sizeof(float);
    return gated_bytes + decode_partials_bytes;
}

int32_t RoutedSwiGluPlugin::enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                                    nvinfer1::PluginTensorDesc const*, void const* const* inputs,
                                    void* const* outputs, void* workspace,
                                    cudaStream_t stream) noexcept {
    if (input_desc == nullptr || inputs == nullptr || outputs == nullptr || workspace == nullptr)
        return -1;
    if (input_desc[0].dims.nbDims != 2 || input_desc[1].dims.nbDims != 2 ||
        input_desc[3].dims.nbDims != 3 || input_desc[4].dims.nbDims != 3 ||
        input_desc[5].dims.nbDims != 3)
        return -1;
    const int32_t rows = input_desc[0].dims.d[0];
    const int32_t hidden_size = input_desc[0].dims.d[1];
    const int32_t top_k = input_desc[1].dims.d[1];
    const int32_t num_experts = input_desc[3].dims.d[0];
    const int32_t intermediate_size = input_desc[3].dims.d[2];
    if (rows <= 0 || rows > max_rows_ || hidden_size <= 0 || top_k <= 0 || num_experts <= 0 ||
        intermediate_size <= 0)
        return -1;
    return launch_routed_swiglu(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
                                outputs[0], workspace, rows, top_k, num_experts, hidden_size,
                                intermediate_size, stream);
}

int32_t launch_routed_swiglu(void const* input, void const* expert_indices,
                             void const* route_weights, void const* gate_weights,
                             void const* up_weights, void const* down_weights, void* output,
                             void* workspace, int32_t rows, int32_t top_k, int32_t num_experts,
                             int32_t hidden_size, int32_t intermediate_size,
                             cudaStream_t stream) noexcept {
    auto* gated = static_cast<__half*>(workspace);
    if (rows == 1) {
        auto* gate_partials =
            reinterpret_cast<float*>(gated + static_cast<size_t>(rows) * top_k * intermediate_size);
        auto* up_partials =
            gate_partials + static_cast<size_t>(top_k) * kDecodeSplitK * intermediate_size;
        const dim3 split_grid(static_cast<unsigned>((intermediate_size + kThreads - 1) / kThreads),
                              static_cast<unsigned>(top_k * kDecodeSplitK));
        routed_gate_up_split_kernel<<<split_grid, kThreads, 0, stream>>>(
            static_cast<__half const*>(input), static_cast<int32_t const*>(expert_indices),
            static_cast<__half const*>(gate_weights), static_cast<__half const*>(up_weights),
            gate_partials, up_partials, top_k, num_experts, hidden_size, intermediate_size);
        const dim3 reduce_grid(static_cast<unsigned>((intermediate_size + kThreads - 1) / kThreads),
                               static_cast<unsigned>(top_k));
        routed_gate_up_reduce_kernel<<<reduce_grid, kThreads, 0, stream>>>(
            gate_partials, up_partials, gated, top_k, intermediate_size);
    } else {
        const dim3 gate_grid(static_cast<unsigned>((intermediate_size + kThreads - 1) / kThreads),
                             static_cast<unsigned>(rows), static_cast<unsigned>(top_k));
        routed_gate_up_kernel<<<gate_grid, kThreads, 0, stream>>>(
            static_cast<__half const*>(input), static_cast<int32_t const*>(expert_indices),
            static_cast<__half const*>(gate_weights), static_cast<__half const*>(up_weights), gated,
            rows, top_k, num_experts, hidden_size, intermediate_size);
    }
    const dim3 down_grid(static_cast<unsigned>((hidden_size + kThreads - 1) / kThreads),
                         static_cast<unsigned>(rows));
    routed_down_kernel<<<down_grid, kThreads, 0, stream>>>(
        gated, static_cast<int32_t const*>(expert_indices),
        static_cast<__half const*>(route_weights), static_cast<__half const*>(down_weights),
        static_cast<__half*>(output), rows, top_k, num_experts, hidden_size, intermediate_size);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

} // namespace trtmc

static nvinfer1::PluginRegistrar<trtmc::RoutedSwiGluCreator> plugin_registrar_routed_swiglu{};

#endif // TRTMC_HAS_TRT
