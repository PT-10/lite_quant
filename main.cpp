#include "safetensor_parser.hpp"
#include "fp_converter.hpp"
#include <iostream>

int main() {
    SafeTensorFile safetensors("Llama-3.2-1B-Instruct/model.safetensors");

    if (!safetensors.valid()) {
        std::cerr << "Failed to parse safetensors\n";
        return 1;
    }

    auto name = safetensors.first_tensor_name();
    std::cout << "First tensor: " << name << "\n";

    auto fp32_data = safetensors.load_tensor_fp32(name);
    std::cout << "Loaded " << fp32_data.size() << " FP32 elements\n";

    std::vector<uint16_t> fp16_data;
    fp16_data.reserve(fp32_data.size());

    for (float x : fp32_data) {
        fp16_data.push_back(fp16::float32_to_fp16(x));
    }

    std::cout << "Converted to FP16, count = " << fp16_data.size() << "\n";
    return 0;
}
