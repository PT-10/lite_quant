#include "fp16_converter.hpp"

namespace fp16 {

uint16_t float32_to_float16(float f) {
    uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp  = (bits >> 23) & 0xFF;
    uint32_t mant = bits & 0x7FFFFF;

    int32_t E_actual = exp - 127;       
    int32_t exp16 = E_actual + 15;      

    if (exp16 <= 0) {
        exp16 = 0; 
        mant   = 0; 
    }
    if (exp16 >= 31) {
        exp16 = 31;
        mant   = 0;
    }

    uint32_t mant16 = mant >> (23 - 10);  
    uint16_t out = (sign << 15) | ((exp16 & 0x1F) << 10) | (mant16 & 0x3FF);
    return out;
}

} // namespace fp16
