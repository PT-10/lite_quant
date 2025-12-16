#include "safetensor_parser.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <nlohmann/json.hpp>  // make sure this header is in include/
using json = nlohmann::json;

SafeTensorFile::SafeTensorFile(const std::string &filename)
    : filename_(filename) {
    ok_ = parse_header();
}

bool SafeTensorFile::valid() const {
    return ok_;
}

bool SafeTensorFile::parse_header() {
    std::ifstream fin(filename_, std::ios::binary);
    if (!fin) {
        std::cerr << "Failed to open " << filename_ << "\n";
        return false;
    }

    // Read 8-byte little-endian header length
    uint64_t header_len;
    fin.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    header_len = le64toh(header_len);

    // Read JSON header
    std::vector<char> buffer(header_len);
    fin.read(buffer.data(), header_len);

    json j;
    try {
        j = json::parse(buffer.begin(), buffer.end());
    } catch (json::parse_error &e) {
        std::cerr << "JSON parse error: " << e.what() << "\n";
        return false;
    }

    for (auto it = j.begin(); it != j.end(); ++it) {
        TensorInfo t;

        // dtype
        auto dtype_it = it.value().find("dtype");
        if (dtype_it != it.value().end() && !dtype_it->is_null())
            t.dtype = dtype_it->get<std::string>();

        // shape
        auto shape_it = it.value().find("shape");
        if (shape_it != it.value().end() && shape_it->is_array())
            t.shape.assign(shape_it->begin(), shape_it->end());

        // data_offsets
        auto offs_it = it.value().find("data_offsets");
        if (offs_it != it.value().end() && offs_it->is_array() && offs_it->size() == 2) {
            t.data_begin = offs_it->at(0).get<uint64_t>();
            t.data_end   = offs_it->at(1).get<uint64_t>();
        } else {
            std::cerr << "Warning: tensor " << it.key() << " has invalid data_offsets, skipping\n";
            continue;
        }

        tensors_.emplace(it.key(), t);
    }

    if (tensors_.empty()) {
        std::cerr << "No valid tensors found in file\n";
        return false;
    }

    return true;
}

std::string SafeTensorFile::first_tensor_name() const {
    if (tensors_.empty()) return {};
    return tensors_.begin()->first;
}

const TensorInfo &SafeTensorFile::get_tensor_info(const std::string &name) const {
    return tensors_.at(name);
}

std::vector<float> SafeTensorFile::load_tensor_fp32(const std::string &name) const {
    const auto &tinfo = get_tensor_info(name);
    std::ifstream fin(filename_, std::ios::binary);

    uint64_t header_len;
    fin.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    header_len = le64toh(header_len);

    fin.seekg(header_len + tinfo.data_begin);

    size_t nbytes = tinfo.data_end - tinfo.data_begin;
    size_t nelems = nbytes / sizeof(float);

    std::vector<float> data(nelems);
    fin.read(reinterpret_cast<char*>(data.data()), nbytes);
    return data;
}
