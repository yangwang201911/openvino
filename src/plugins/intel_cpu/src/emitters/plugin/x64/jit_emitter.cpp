// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_emitter.hpp"

#include <xbyak/xbyak.h>

#include <algorithm>
#include <cassert>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <vector>

#include "emitters/utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"

using namespace dnnl::impl::cpu;
using namespace dnnl::impl;
using namespace Xbyak;

namespace ov::intel_cpu {

size_t jit_emitter::get_max_vecs_count() const {
    return one_of(host_isa_, cpu::x64::avx512_core, cpu::x64::avx512_core) ? 32 : 16;
}

size_t jit_emitter::get_vec_length() const {
    if (host_isa_ == cpu::x64::avx512_core) {
        return 64;
    }
    if (host_isa_ == cpu::x64::avx2) {
        return 32;
    }
    return 16;
}

void jit_emitter::push_vec(const Xbyak::Address& addr, size_t vec_idx) const {
    if (host_isa_ == cpu::x64::sse41) {
        h->uni_vmovups(addr, Xmm(vec_idx));
    } else if (host_isa_ == cpu::x64::avx2) {
        h->uni_vmovups(addr, Ymm(vec_idx));
    } else {
        h->uni_vmovups(addr, Zmm(vec_idx));
    }
}

void jit_emitter::pop_vec(size_t vec_idx, const Xbyak::Address& addr) const {
    if (host_isa_ == cpu::x64::sse41) {
        h->uni_vmovups(Xmm(vec_idx), addr);
    } else if (host_isa_ == cpu::x64::avx2) {
        h->uni_vmovups(Ymm(vec_idx), addr);
    } else {
        h->uni_vmovups(Zmm(vec_idx), addr);
    }
}

size_t jit_emitter::aux_vecs_count() const {
    return 0;
}

emitter_in_out_map jit_emitter::get_in_out_type() const {
    return in_out_type_;
}

size_t jit_emitter::aux_gprs_count() const {
    // We need one gpr to load table address
    return entry_map_.empty() ? 0 : 1;
}

std::set<std::vector<element::Type>> jit_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {};
}

void jit_emitter::emitter_preamble(const std::vector<size_t>& in_idxs,
                                   const std::vector<size_t>& out_idxs,
                                   const std::vector<size_t>& pool_vec_idxs,
                                   const std::vector<size_t>& pool_gpr_idxs) const {
    using namespace Xbyak::util;
    bool is_vec_input =
        (in_out_type_ == emitter_in_out_map::vec_to_vec) || (in_out_type_ == emitter_in_out_map::vec_to_gpr);
    bool is_vec_output =
        (in_out_type_ == emitter_in_out_map::vec_to_vec) || (in_out_type_ == emitter_in_out_map::gpr_to_vec);

    for (auto idx : pool_vec_idxs) {
        aux_vec_idxs.push_back(idx);
    }

    // For sse41 mask register has to be Xmm(0)
    if (host_isa_ == cpu::x64::sse41 && aux_vecs_count() > 0) {
        size_t idx = 0;
        if (is_vec_input) {
            OV_CPU_JIT_EMITTER_ASSERT(std::find(in_idxs.begin(), in_idxs.end(), idx) == in_idxs.end(),
                                      "Xmm(0) cannot be input register in SSE41");
        }
        if (is_vec_output) {
            OV_CPU_JIT_EMITTER_ASSERT(std::find(out_idxs.begin(), out_idxs.end(), idx) == out_idxs.end(),
                                      "Xmm(0) cannot be output register in SSE41");
        }
        if (std::find(aux_vec_idxs.begin(), aux_vec_idxs.end(), idx) == aux_vec_idxs.end()) {
            aux_vec_idxs.push_back(idx);
            preserved_vec_idxs.push_back(idx);
        }

        // moving mask vector at the beginning of aux vectors list to simplify further processing
        for (size_t& aux_vec_idx : aux_vec_idxs) {
            if (aux_vec_idx == 0) {
                size_t tmp = aux_vec_idxs[0];
                aux_vec_idxs[0] = aux_vec_idx;
                aux_vec_idx = tmp;
                break;
            }
        }
    }

    for (size_t idx = 0; idx < get_max_vecs_count(); idx++) {
        if (aux_vec_idxs.size() >= aux_vecs_count()) {
            break;
        }

        if (is_vec_input) {
            if (std::find(in_idxs.begin(), in_idxs.end(), idx) != in_idxs.end()) {
                continue;
            }
        }
        if (is_vec_output) {
            if (std::find(out_idxs.begin(), out_idxs.end(), idx) != out_idxs.end()) {
                continue;
            }
        }
        if (std::find(aux_vec_idxs.begin(), aux_vec_idxs.end(), idx) != aux_vec_idxs.end()) {
            continue;
        }
        if (std::find(preserved_vec_idxs.begin(), preserved_vec_idxs.end(), idx) != preserved_vec_idxs.end()) {
            continue;
        }

        aux_vec_idxs.push_back(idx);
        preserved_vec_idxs.push_back(idx);
    }
    if (aux_vec_idxs.size() < aux_vecs_count()) {
        OV_CPU_JIT_EMITTER_THROW("Failed to allocate required number of vector registers");
    }

    // Same logic but to allocate gprs
    for (auto idx : pool_gpr_idxs) {
        aux_gpr_idxs.push_back(idx);
    }

    for (size_t gpr_idx = 0; gpr_idx <= Operand::R15; ++gpr_idx) {
        size_t _idx = Operand::R15 - gpr_idx;  // we allocate from the end

        if (aux_gpr_idxs.size() >= aux_gprs_count()) {
            break;
        }
        if (_idx == Operand::RSP) {
            continue;
        }
        if (!is_vec_input) {
            if (std::find(in_idxs.begin(), in_idxs.end(), _idx) != in_idxs.end()) {
                continue;
            }
        }
        if (!is_vec_output) {
            if (std::find(out_idxs.begin(), out_idxs.end(), _idx) != out_idxs.end()) {
                continue;
            }
        }
        if (std::find(aux_gpr_idxs.begin(), aux_gpr_idxs.end(), _idx) != aux_gpr_idxs.end()) {
            continue;
        }
        if (std::find(preserved_gpr_idxs.begin(), preserved_gpr_idxs.end(), _idx) != preserved_gpr_idxs.end()) {
            continue;
        }

        aux_gpr_idxs.push_back(_idx);
        preserved_gpr_idxs.push_back(_idx);
    }
    if (aux_gpr_idxs.size() < aux_gprs_count()) {
        OV_CPU_JIT_EMITTER_THROW("Failed to allocate required number of general-purpose registers");
    }

    if (!entry_map_.empty()) {
        // last aux_gpr_idx is for p_table, we can use aux_gpr_idxs from idx 0 for other purpose
        OPENVINO_ASSERT(!aux_gpr_idxs.empty(), "No aux gprs available");
        p_table = Reg64(aux_gpr_idxs[aux_gprs_count() - 1]);
        aux_gpr_idxs.erase(aux_gpr_idxs.end() - 1);
    }

    for (uint64_t preserved_gpr_idx : preserved_gpr_idxs) {
        h->push(Reg64(preserved_gpr_idx));
    }

    if (!preserved_vec_idxs.empty()) {
        h->sub(h->rsp, preserved_vec_idxs.size() * get_vec_length());
    }

    for (size_t i = 0; i < preserved_vec_idxs.size(); ++i) {
        push_vec(h->ptr[h->rsp + i * get_vec_length()], preserved_vec_idxs[i]);
    }

    if (!entry_map_.empty()) {
        load_table_addr();
    }
}

void jit_emitter::emitter_postamble() const {
    using namespace Xbyak::util;

    for (size_t i = 0; i < preserved_vec_idxs.size(); ++i) {
        pop_vec(preserved_vec_idxs[i], h->ptr[h->rsp + i * get_vec_length()]);
    }

    if (!preserved_vec_idxs.empty()) {
        h->add(h->rsp, preserved_vec_idxs.size() * get_vec_length());
    }

    for (int i = preserved_gpr_idxs.size() - 1; i >= 0; --i) {
        h->pop(Reg64(preserved_gpr_idxs[i]));
    }

    preserved_vec_idxs.clear();
    preserved_gpr_idxs.clear();

    aux_vec_idxs.clear();
    aux_gpr_idxs.clear();
}

void jit_emitter::emit_data() const {
    h->align(64);
    h->L(*l_table);

    // Assumption: entries can be inserted with dd, so they should be 4 bytes.
    static_assert(sizeof(table_entry_val_t) == 4);

    // Run through the map and insert values stored there
    for (const auto& it : entry_map_) {
        const auto& te = it.second;  // get map entry for a given key
        const auto len = te.bcast ? get_vec_length() : sizeof(table_entry_val_t);
        for (size_t d = 0; d < len; d += sizeof(table_entry_val_t)) {
            h->dd(te.val);
        }
    }
}

void jit_emitter::prepare_table() {
    register_table_entries();

    // Now that we registered the entries, we set the offsets.  No
    // entries should be registered after this point.  This allows to
    // expect the same order when injecting the table entries in
    // prepare_table.
    size_t off = 0;
    for (auto& it : entry_map_) {
        auto& te = it.second;
        te.off = off;
        off += te.bcast ? get_vec_length() : sizeof(table_entry_val_t);
    }
}

void jit_emitter::emit_code_impl(const std::vector<size_t>& in_idxs,
                                 const std::vector<size_t>& out_idxs,
                                 const std::vector<size_t>& pool_vec_idxs,
                                 const std::vector<size_t>& pool_gpr_idxs) const {
    emitter_preamble(in_idxs, out_idxs, pool_vec_idxs, pool_gpr_idxs);

    emit_impl(in_idxs, out_idxs);

    emitter_postamble();
}

}  // namespace ov::intel_cpu
