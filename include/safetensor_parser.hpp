#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <map>

struct TensorInfo {
    std::string dtype = "F32";       // default dtype
    std::vector<int64_t> shape;
    uint64_t data_begin = 0;
    uint64_t data_end = 0;
};

class SafeTensorFile {
public:
    explicit SafeTensorFile(const std::string &filename);

    bool valid() const;
    std::string first_tensor_name() const;
    const TensorInfo &get_tensor_info(const std::string &name) const;
    std::vector<float> load_tensor_fp32(const std::string &name) const;

private:
    bool parse_header();
    std::string filename_;
    std::map<std::string, TensorInfo> tensors_;
    bool ok_ = false;
};
