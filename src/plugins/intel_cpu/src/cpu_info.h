// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/x64/xbyak/xbyak_util.h>

#include <cpu/x64/amx_tile_configure.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <mutex>
#include <string>
using Xbyak::Tmm;
using Xbyak::Xmm;
using Xbyak::Ymm;
using Xbyak::Zmm;
using Xbyak::util::Cpu;
#include <cpu/platform.hpp>

#include "openvino/core/type/element_type.hpp"

#define MAX_CPUS 64

namespace ov {
namespace intel_cpu {
extern float runtimeFreq;
float get_runtime_freq(int core_id = 0);
class CPUInfo {
public:
    CPUInfo();
    float getPeakGOPSImpl(ov::element::Type precision);
    void printDetails();

private:
    enum class ISA {
        sse,
        sse2,
        sse3,
        ssse3,
        sse4_1,
        sse4_2,
        avx,
        avx2,
        fma,
        avx512_common,
        avx512_core,
        avx512_mic,
        avx512_mic_4ops,
        avx512_vnni,
        avx512_fp16,
        amx_int8,
        amx_bf16,
    };

    void init();
    bool checkIsaSupport(ISA cpu_isa) {
        switch (cpu_isa) {
        case ISA::sse:
            return cpu.has(Cpu::tSSE);
        case ISA::sse2:
            return cpu.has(Cpu::tSSE2);
        case ISA::sse3:
            return cpu.has(Cpu::tSSE3);
        case ISA::ssse3:
            return cpu.has(Cpu::tSSSE3);
        case ISA::sse4_1:
            return cpu.has(Cpu::tSSE41);
        case ISA::sse4_2:
            return cpu.has(Cpu::tSSE42);
        case ISA::avx:
            return cpu.has(Cpu::tAVX);
        case ISA::avx2:
            return cpu.has(Cpu::tAVX2);
        case ISA::fma:
            return cpu.has(Cpu::tFMA);
        case ISA::avx512_common:
            return cpu.has(Cpu::tAVX512F);
        case ISA::avx512_core:
            return cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512BW) && cpu.has(Cpu::tAVX512VL) &&
                   cpu.has(Cpu::tAVX512DQ);
        case ISA::avx512_mic:
            return cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512CD) && cpu.has(Cpu::tAVX512ER) &&
                   cpu.has(Cpu::tAVX512PF);
        case ISA::avx512_mic_4ops:
            return checkIsaSupport(ISA::avx512_mic) && cpu.has(Cpu::tAVX512_4FMAPS) && cpu.has(Cpu::tAVX512_4VNNIW);
        case ISA::avx512_vnni:
            return cpu.has(Cpu::tAVX512F);  //&&
                                            // cpu.has(Cpu::tAVX512_VNNI);
        case ISA::avx512_fp16:
            return cpu.has(Cpu::tAVX512_FP16);
        case ISA::amx_bf16:
            return cpu.has(Cpu::tAMX_BF16) && dnnl::impl::cpu::x64::amx::is_available();
        case ISA::amx_int8:
            return cpu.has(Cpu::tAMX_INT8) && dnnl::impl::cpu::x64::amx::is_available();
        }

        return false;
    }
    bool haveSSE() {
        return have_sse;
    }
    bool haveSSEX() {
        return have_sse2 && have_ssse3 && have_sse4_1 && have_sse4_2;
    }
    bool haveAVX() {
        return have_avx;
    }
    bool haveAVX2() {
        return have_avx2;
    }
    bool haveAVX512() {
        return have_avx512f;
    }
    bool haveAVX512FP16() {
        return have_avx512_fp16;
    }
    bool haveAMXBF16() {
        return have_amx_bf16;
    }
    bool haveAMXINT8() {
        return have_amx_int8;
    }

    float calcComputeBlockIPC(ov::element::Type precision);

    float getFrequency(const std::string path);
    float getMaxCPUFreq(size_t core_id);
    float getMinCPUFreq(size_t core_id);
    bool isFrequencyFixed();

#ifdef WIN32
    float currGHz = 1.0f;
#endif

    bool have_sse = false;
    bool have_sse2 = false;
    bool have_ssse3 = false;
    bool have_sse4_1 = false;
    bool have_sse4_2 = false;
    bool have_avx = false;
    bool have_avx2 = false;
    bool have_fma = false;
    bool have_avx512f = false;
    bool have_avx512_fp16 = false;
    bool have_vnni = false;
    bool have_amx_int8 = false;
    bool have_amx_bf16 = false;

    // Micro architecture level
    uint32_t simd_size = 1;
    float instructions_per_cycle = 1.0f;
    uint32_t operations_per_instruction = 1;

    // Machine architecture level
    float freqGHz = 1.0f;
    uint32_t cores_per_socket = 1;
    uint32_t sockets_per_node = 1;
    std::string ISA_detailed;
    Xbyak::util::Cpu cpu;
};

template <typename T>
struct RegMap;

template <>
struct RegMap<Xbyak::Xmm> {
    void save(Xbyak::CodeGenerator* g, int idx, int off) {
        g->movaps(g->ptr[g->rsp + off], Xmm(idx));
    }

    void restore(Xbyak::CodeGenerator* g, int idx, int off) {
        g->movaps(Xmm(idx), g->ptr[g->rsp + off]);
    }

    void killdep(Xbyak::CodeGenerator* g, int idx) {
        g->xorps(Xmm(idx), Xmm(idx));
    }
};

template <>
struct RegMap<Xbyak::Ymm> {
    void save(Xbyak::CodeGenerator* g, int idx, int off) {
        g->vmovaps(g->ptr[g->rsp + off], Ymm(idx));
    }

    void restore(Xbyak::CodeGenerator* g, int idx, int off) {
        g->vmovaps(Ymm(idx), g->ptr[g->rsp + off]);
    }
    void killdep(Xbyak::CodeGenerator* g, int idx) {
        g->vxorps(Ymm(idx), Ymm(idx), Ymm(idx));
    }
};

template <>
struct RegMap<Xbyak::Zmm> {
    void save(Xbyak::CodeGenerator* g, int idx, int off) {
        g->vmovaps(g->ptr[g->rsp + off], Zmm(idx));
    }

    void restore(Xbyak::CodeGenerator* g, int idx, int off) {
        g->vmovaps(Zmm(idx), g->ptr[g->rsp + off]);
    }

    void killdep(Xbyak::CodeGenerator* g, int idx) {
        g->vpxorq(Zmm(idx), Zmm(idx), Zmm(idx));
    }
};

template <>
struct RegMap<Xbyak::Tmm> {
    RegMap<Xbyak::Tmm>() {
#if DNNL_X64
        dnnl::impl::cpu::x64::palette_config_t tconf = {0};
        auto palette_id = dnnl::impl::cpu::x64::amx::get_target_palette();
        tconf.palette_id = palette_id;
        for (int index = 0; index < 8; index++) {
            tconf.rows[index] = 16;  // dnnl::impl::cpu::x64::amx::get_max_rows(tconf.palette_id);
        }
        for (int index = 0; index < 8; index++) {
            tconf.cols[index] = 64;  // dnnl::impl::cpu::x64::amx::get_max_column_bytes(tconf.palette_id);
        }
        // configura AMX tiles
        dnnl::impl::cpu::x64::amx_tile_configure((const char*)&tconf);
#endif
    }

    ~RegMap<Xbyak::Tmm>() {
#if DNNL_X64
        dnnl::impl::cpu::x64::amx_tile_release();
#endif
    }

    void save(Xbyak::CodeGenerator* g, int idx, int off) {
#if DNNL_X64
        // No need to save AMX registers here and juse clean registers.
        idx = idx % 4;
        g->tilezero(Tmm(idx));
#endif
    }

    void restore(Xbyak::CodeGenerator* g, int idx, int off) {
#if DNNL_X64
        // No need to restore AMX registers here and juse clean registers.
        idx = idx % 4;
        g->tilezero(Tmm(idx));
#endif
    }

    void killdep(Xbyak::CodeGenerator* g, int idx) {
        return;
    }
};

template <typename RegType, typename Gen, typename F>
struct ThroughputGenerator {
    void operator()(Gen* g, RegMap<RegType>& rm, F f, int num_insn) {
        for (int j = 0; j < num_insn / 12; j++) {
            runtimeFreq += get_runtime_freq();
            runtimeFreq = runtimeFreq / (j + 1);
            for (int i = 0; i < 12; i++) {
                f(g, 4 + i, 4 + i);
            }
        }
    }
};

template <typename RegType, typename F>
struct Generator : public Xbyak::CodeGenerator {
    Generator(F f, int num_loop, int num_insn) {
        RegMap<RegType> rm;

        int reg_size = 64;
        int num_reg = 12;

        push(rbp);
        mov(rbp, rsp);
        and_(rsp, -(Xbyak::sint64)64);
        sub(rsp, reg_size * (num_reg + 1));

        for (int i = 0; i < num_reg; i++)
            rm.save(this, 4 + i, -reg_size * (12 - i));

        for (int i = 0; i < num_reg; i++)
            rm.killdep(this, 4 + i);

        mov(rcx, num_loop);

        align(16);
        L("@@");
        ThroughputGenerator<RegType, Generator, F>()(this, rm, f, num_insn);
        dec(rcx);
        jnz("@b");

        for (int i = 0; i < num_reg; i++)
            rm.restore(this, 4 + i, -reg_size * (12 - i));

        mov(rsp, rbp);
        pop(rbp);
        ret();
    }
};
}  // namespace intel_cpu
}  // namespace ov