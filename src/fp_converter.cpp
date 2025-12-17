#include "fp_converter.hpp"

#include <cstdint>
#include <cstring>
namespace fp16 {

uint16_t float32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));

    uint32_t sign = (x >> 31) & 1;
    int32_t  exp  = (x >> 23) & 0xFF;
    uint32_t mant = x & 0x7FFFFF;

    // NaN or Inf
    if (exp == 255) {
        if (mant) {
            return (sign << 15) | 0x7E00; // canonical NaN
        }
        return (sign << 15) | 0x7C00;     // Inf
    }

    int32_t e = exp - 127 + 15;

    // Overflow to Inf
    if (e >= 31) {
        return (sign << 15) | 0x7C00;
    }

    // Subnormal or zero
    if (e <= 0) {
        if (e < -10) {
            return sign << 15; // underflow to zero
        }

        mant |= 0x800000; // implicit 1
        uint32_t shift = 14 - e;
        uint32_t mant16 = mant >> shift;

        // RNE rounding
        if ((mant >> (shift - 1)) & 1) {
            mant16 += 1;
        }

        return (sign << 15) | (mant16 & 0x3FF);
    }

    // Normalized
    uint32_t mant16 = mant >> 13;
    if (mant & 0x1000) { // RNE
        mant16 += 1;
        if (mant16 == 0x400) {
            mant16 = 0;
            e += 1;
            if (e == 31) {
                return (sign << 15) | 0x7C00;
            }
        }
    }

    return (sign << 15) | (e << 10) | (mant16 & 0x3FF);
}
}