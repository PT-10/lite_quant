#pragma once
#include <cstdint>

namespace fp16 {
    uint16_t float32_to_fp16(float f);
}

namespace bf16 {
    uint16_t float32_to_bf16(float f);
}

namespace fp8_e4m3 {
    uint8_t float32_to_fp8(float f);
    uint8_t bf16_to_fp8(uint16_t bf);
    uint8_t fp16_to_fp8(uint16_t h);
}

namespace fp8_e5m2 {
    uint8_t float32_to_fp8(float f);
    uint8_t bf16_to_fp8(uint16_t bf);
    uint8_t fp16_to_fp8(uint16_t h);
}
