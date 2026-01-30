/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef HAMMING_AVX512_INL_H
#define HAMMING_AVX512_INL_H

// AVX512 version
// The _mm512_popcnt_epi64 intrinsic is used to accelerate Hamming distance
// calculations in HammingComputerDefault and HammingComputer64. This intrinsic
// is not available in the default Faiss avx512 build mode but is only
// available in the avx512_spr build mode, which targets Intel(R) Sapphire
// Rapids.

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <faiss/impl/platform_macros.h>

#include <immintrin.h>

#include <cstdint>
#include <cstdio>
#include <iostream>


namespace faiss {

/* Elementary Hamming distance computation: unoptimized  */
template <size_t nbits, typename T>
inline T hamming(const uint8_t* bs1, const uint8_t* bs2) {
    const size_t nbytes = nbits / 8;
    size_t i;
    T h = 0;
    for (i = 0; i < nbytes; i++) {
        h += (T)hamdis_tab_ham_bytes[bs1[i] ^ bs2[i]];
    }
    return h;
}

/* Hamming distances for multiples of 64 bits */
template <size_t nbits>
inline hamdis_t hamming(const uint64_t* bs1, const uint64_t* bs2) {
    const size_t nwords = nbits / 64;
    size_t i;
    hamdis_t h = 0;
    for (i = 0; i < nwords; i++) {
        h += popcount64(bs1[i] ^ bs2[i]);
    }
    return h;
}

/* specialized (optimized) functions */
template <>
inline hamdis_t hamming<64>(const uint64_t* pa, const uint64_t* pb) {
    return popcount64(pa[0] ^ pb[0]);
}

template <>
inline hamdis_t hamming<128>(const uint64_t* pa, const uint64_t* pb) {
    return popcount64(pa[0] ^ pb[0]) + popcount64(pa[1] ^ pb[1]);
}

template <>
inline hamdis_t hamming<256>(const uint64_t* pa, const uint64_t* pb) {
    return popcount64(pa[0] ^ pb[0]) + popcount64(pa[1] ^ pb[1]) +
            popcount64(pa[2] ^ pb[2]) + popcount64(pa[3] ^ pb[3]);
}

/* Hamming distances for multiple of 64 bits */
inline hamdis_t hamming(
        const uint64_t* bs1,
        const uint64_t* bs2,
        size_t nwords) {
    hamdis_t h = 0;
    for (size_t i = 0; i < nwords; i++) {
        h += popcount64(bs1[i] ^ bs2[i]);
    }
    return h;
}


// ---- helpers ----
static inline uint32_t popcnt64_scalar(uint64_t x) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
    return (uint32_t)__popcnt64(x);
#else
    return (uint32_t)__builtin_popcountll((unsigned long long)x);
#endif
}

// popcount of an arbitrary-length bitmap in bytes (handles 4096 bits = 512 bytes too)
static inline int total_popcnt_bytes_avx512(const uint8_t* a8, int nbytes) {
    const uint64_t* a64 = reinterpret_cast<const uint64_t*>(a8);
    int n_words = nbytes / 8;
    int rem     = nbytes % 8;
    int sum = 0;

#if defined(__AVX512VPOPCNTDQ__)
    int i = 0;
    int blocks = n_words / 8;               // 8x u64 per 512-bit block
    for (; i < blocks; ++i) {
        __m512i va = _mm512_loadu_si512((const void*)&a64[i*8]);
        __m512i pc = _mm512_popcnt_epi64(va);
        alignas(64) uint64_t tmp[8];
        _mm512_store_si512((__m512i*)tmp, pc);
        sum += int(tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
    }
    int off = blocks * 8;
    for (int j = off; j < n_words; ++j) sum += popcount64(a64[j]);
#else
    for (int j = 0; j < n_words; ++j) sum += popcount64(a64[j]);
#endif
    // byte tail (rare for 4096-bit bitmaps) xxx xxx
    if (rem) {
        const uint8_t* p = a8 + 8 * n_words;
        static const uint8_t lut[256] = {
            #define B2(n) n, n+1, n+1, n+2
            #define B4(n) B2(n), B2(n+1), B2(n+1), B2(n+2)
            #define B6(n) B4(n), B4(n+1), B4(n+1), B4(n+2)
            B6(0), B6(1), B6(1), B6(2)
            #undef B6
            #undef B4
            #undef B2
        };
        for (int k = 0; k < rem; ++k) sum += lut[p[k]];
    }
    return sum;
}

static inline uint32_t popcount32_u(uint32_t x) {
#if defined(_MSC_VER)
    return (uint32_t)__popcnt(x);
#else
    return (uint32_t)__builtin_popcount(x);
#endif
}



static inline int hsum8_u64(__m512i v) {
    alignas(64) uint64_t t[8];
    _mm512_store_si512((__m512i*)t, v);
    return int(t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7]);
}



/******************************************************************
 * The HammingComputer series of classes compares a single code of
 * size 4 to 32 to incoming codes. They are intended for use as a
 * template class where it would be inefficient to switch on the code
 * size in the inner loop. Hopefully the compiler will inline the
 * hamming() functions and put the a0, a1, ... in registers.
 ******************************************************************/

struct HammingComputer4 {
    uint32_t a0 = 0;
    int pa_cached = 0;

    HammingComputer4() {}
    HammingComputer4(const uint8_t* a, int code_size) { set(a, code_size); }

    void set(const uint8_t* a, int code_size) {
        assert(code_size == 4);
        memcpy(&a0, a, 4);
        pa_cached = (int)popcount32_u(a0); // or popcount32(a0)
    }

    inline int hamming(const uint8_t* b) const {
        uint32_t b0;
        memcpy(&b0, b, 4);
        return (int)popcount32_u(b0 ^ a0);
    }

    inline float jaccard_with_pb(const uint8_t* b, int pb) const {
        const int px = hamming(b);
        const int num = (pa_cached + pb - px);
        const int den = (pa_cached + pb + px);
        return den ? float(num) / float(den) : 1.0f;
    }

    inline float jaccard_onthefly(const uint8_t* b) const {
        uint32_t b0;
        memcpy(&b0, b, 4);
        const int pb = (int)popcount32_u(b0);
        const int px = (int)popcount32_u(b0 ^ a0);
        const int num = (pa_cached + pb - px);
        const int den = (pa_cached + pb + px);
        return den ? float(num) / float(den) : 1.0f;
    }

    inline static constexpr int get_code_size() { return 4; }
};


struct HammingComputer8 {
    uint64_t a0 = 0;
    int pa_cached = 0;

    HammingComputer8() {}
    HammingComputer8(const uint8_t* a, int code_size) { set(a, code_size); }

    void set(const uint8_t* a, int code_size) {
        assert(code_size == 8);
        memcpy(&a0, a, 8);
        pa_cached = (int)popcount64(a0);
    }

    inline int hamming(const uint8_t* b) const {
        uint64_t b0;
        memcpy(&b0, b, 8);
        return (int)popcount64(b0 ^ a0);
    }

    inline float jaccard_with_pb(const uint8_t* b, int pb) const {
        const int px = hamming(b);
        const int num = (pa_cached + pb - px);
        const int den = (pa_cached + pb + px);
        return den ? float(num) / float(den) : 1.0f;
    }

    inline float jaccard_onthefly(const uint8_t* b) const {
        uint64_t b0;
        memcpy(&b0, b, 8);
        const int pb = (int)popcount64(b0);
        const int px = (int)popcount64(b0 ^ a0);
        const int num = (pa_cached + pb - px);
        const int den = (pa_cached + pb + px);
        return den ? float(num) / float(den) : 1.0f;
    }

    inline static constexpr int get_code_size() { return 8; }
};


struct HammingComputer16 {
    uint64_t a0 = 0, a1 = 0;
    int pa_cached = 0;

    HammingComputer16() {}
    HammingComputer16(const uint8_t* a8, int code_size) { set(a8, code_size); }

    void set(const uint8_t* a8, int code_size) {
        assert(code_size == 16);
        memcpy(&a0, a8 + 0, 8);
        memcpy(&a1, a8 + 8, 8);
        pa_cached = (int)(popcount64(a0) + popcount64(a1));
    }

    inline int hamming(const uint8_t* b8) const {
        uint64_t b0, b1;
        memcpy(&b0, b8 + 0, 8);
        memcpy(&b1, b8 + 8, 8);
        return (int)(popcount64(b0 ^ a0) + popcount64(b1 ^ a1));
    }

    inline float jaccard_with_pb(const uint8_t* b8, int pb) const {
        const int px = hamming(b8);
        const int num = (pa_cached + pb - px);
        const int den = (pa_cached + pb + px);
        return den ? float(num) / float(den) : 1.0f;
    }

    inline float jaccard_onthefly(const uint8_t* b8) const {
        uint64_t b0, b1;
        memcpy(&b0, b8 + 0, 8);
        memcpy(&b1, b8 + 8, 8);
        const int pb = (int)(popcount64(b0) + popcount64(b1));
        const int px = (int)(popcount64(b0 ^ a0) + popcount64(b1 ^ a1));
        const int num = (pa_cached + pb - px);
        const int den = (pa_cached + pb + px);
        return den ? float(num) / float(den) : 1.0f;
    }

    inline static constexpr int get_code_size() { return 16; }
};


// when applied to an array, 1/2 of the 64-bit accesses are unaligned.
// This incurs a penalty of ~10% wrt. fully aligned accesses.
struct HammingComputer20 {
    uint64_t a0 = 0, a1 = 0;
    uint32_t a2 = 0;
    int pa_cached = 0;

    HammingComputer20() {}
    HammingComputer20(const uint8_t* a8, int code_size) { set(a8, code_size); }

    void set(const uint8_t* a8, int code_size) {
        assert(code_size == 20);
        memcpy(&a0, a8 + 0, 8);
        memcpy(&a1, a8 + 8, 8);
        memcpy(&a2, a8 + 16, 4);
        pa_cached = (int)(popcount64(a0) + popcount64(a1) + popcount32_u(a2));
    }

    inline int hamming(const uint8_t* b8) const {
        uint64_t b0, b1;
        uint32_t b2;
        memcpy(&b0, b8 + 0, 8);
        memcpy(&b1, b8 + 8, 8);
        memcpy(&b2, b8 + 16, 4);
        return (int)(popcount64(b0 ^ a0) + popcount64(b1 ^ a1) + popcount32_u(b2 ^ a2));
    }

    inline float jaccard_with_pb(const uint8_t* b8, int pb) const {
        const int px = hamming(b8);
        const int num = (pa_cached + pb - px);
        const int den = (pa_cached + pb + px);
        return den ? float(num) / float(den) : 1.0f;
    }

    inline float jaccard_onthefly(const uint8_t* b8) const {
        uint64_t b0, b1;
        uint32_t b2;
        memcpy(&b0, b8 + 0, 8);
        memcpy(&b1, b8 + 8, 8);
        memcpy(&b2, b8 + 16, 4);
        const int pb = (int)(popcount64(b0) + popcount64(b1) + popcount32_u(b2));
        const int px = (int)(popcount64(b0 ^ a0) + popcount64(b1 ^ a1) + popcount32_u(b2 ^ a2));
        const int num = (pa_cached + pb - px);
        const int den = (pa_cached + pb + px);
        return den ? float(num) / float(den) : 1.0f;
    }

    inline static constexpr int get_code_size() { return 20; }
};


struct HammingComputer32 {
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    int pa_cached = 0;

    HammingComputer32() {}
    HammingComputer32(const uint8_t* a8, int code_size) { set(a8, code_size); }

    void set(const uint8_t* a8, int code_size) {
        assert(code_size == 32);
        memcpy(&a0, a8 + 0, 8);
        memcpy(&a1, a8 + 8, 8);
        memcpy(&a2, a8 + 16, 8);
        memcpy(&a3, a8 + 24, 8);
        pa_cached = (int)(popcount64(a0) + popcount64(a1) + popcount64(a2) + popcount64(a3));
    }

    inline int hamming(const uint8_t* b8) const {
        uint64_t b0, b1, b2, b3;
        memcpy(&b0, b8 + 0, 8);
        memcpy(&b1, b8 + 8, 8);
        memcpy(&b2, b8 + 16, 8);
        memcpy(&b3, b8 + 24, 8);
        return (int)(popcount64(b0 ^ a0) + popcount64(b1 ^ a1) +
                     popcount64(b2 ^ a2) + popcount64(b3 ^ a3));
    }

    inline float jaccard_with_pb(const uint8_t* b8, int pb) const {
        const int px = hamming(b8);
        const int num = (pa_cached + pb - px);
        const int den = (pa_cached + pb + px);
        return den ? float(num) / float(den) : 1.0f;
    }

    inline float jaccard_onthefly(const uint8_t* b8) const {
        uint64_t b0, b1, b2, b3;
        memcpy(&b0, b8 + 0, 8);
        memcpy(&b1, b8 + 8, 8);
        memcpy(&b2, b8 + 16, 8);
        memcpy(&b3, b8 + 24, 8);
        const int pb = (int)(popcount64(b0) + popcount64(b1) + popcount64(b2) + popcount64(b3));
        const int px = (int)(popcount64(b0 ^ a0) + popcount64(b1 ^ a1) +
                             popcount64(b2 ^ a2) + popcount64(b3 ^ a3));
        const int num = (pa_cached + pb - px);
        const int den = (pa_cached + pb + px);
        return den ? float(num) / float(den) : 1.0f;
    }

    inline static constexpr int get_code_size() { return 32; }
};


struct HammingComputer64 {
    uint64_t a0, a1, a2, a3, a4, a5, a6, a7;
    const uint64_t* a = nullptr;
    int pa_cached = 0;

    HammingComputer64() {}
    HammingComputer64(const uint8_t* a8, int code_size) { set(a8, code_size); }

    void set(const uint8_t* a8, int code_size) {
        assert(code_size == 64);
        a = (const uint64_t*)a8;
        a0 = a[0]; a1 = a[1]; a2 = a[2]; a3 = a[3];
        a4 = a[4]; a5 = a[5]; a6 = a[6]; a7 = a[7];
        pa_cached = (int)(popcount64(a0)+popcount64(a1)+popcount64(a2)+popcount64(a3)+
                          popcount64(a4)+popcount64(a5)+popcount64(a6)+popcount64(a7));
    }

    inline int hamming(const uint8_t* b8) const {
        const uint64_t* b = (const uint64_t*)b8;
    #ifdef __AVX512VPOPCNTDQ__
        __m512i vxor = _mm512_xor_si512(_mm512_loadu_si512(a), _mm512_loadu_si512(b));
        __m512i vpc  = _mm512_popcnt_epi64(vxor);
        return _mm512_reduce_add_epi32(vpc);
    #else
        return (int)(popcount64(b[0]^a0)+popcount64(b[1]^a1)+popcount64(b[2]^a2)+popcount64(b[3]^a3)+
                     popcount64(b[4]^a4)+popcount64(b[5]^a5)+popcount64(b[6]^a6)+popcount64(b[7]^a7));
    #endif
    }

    inline float jaccard_with_pb(const uint8_t* b8, int pb) const {
        const int px = hamming(b8);
        const int num = (pa_cached + pb - px);
        const int den = (pa_cached + pb + px);
        return den ? float(num) / float(den) : 1.0f;
    }

    inline float jaccard_onthefly(const uint8_t* b8) const {
        const uint64_t* b = (const uint64_t*)b8;
        int pb = 0, px = 0;
    #ifdef __AVX512VPOPCNTDQ__
        __m512i vb   = _mm512_loadu_si512(b);
        __m512i va   = _mm512_loadu_si512(a);
        __m512i vxor = _mm512_xor_si512(va, vb);
        pb = _mm512_reduce_add_epi32(_mm512_popcnt_epi64(vb));
        px = _mm512_reduce_add_epi32(_mm512_popcnt_epi64(vxor));
    #else
        pb = (int)(popcount64(b[0])+popcount64(b[1])+popcount64(b[2])+popcount64(b[3])+
                   popcount64(b[4])+popcount64(b[5])+popcount64(b[6])+popcount64(b[7]));
        px = (int)(popcount64(b[0]^a0)+popcount64(b[1]^a1)+popcount64(b[2]^a2)+popcount64(b[3]^a3)+
                   popcount64(b[4]^a4)+popcount64(b[5]^a5)+popcount64(b[6]^a6)+popcount64(b[7]^a7));
    #endif
        const int num = (pa_cached + pb - px);
        const int den = (pa_cached + pb + px);
        return den ? float(num) / float(den) : 1.0f;
    }

    inline static constexpr int get_code_size() { return 64; }
};

struct HammingComputerDefault {
    const uint8_t* a8 = nullptr;
    int quotient8 = 0, remainder8 = 0;
    int pa_cached = 0;                 // <-- cached once per query
    HammingComputerDefault() {}
    HammingComputerDefault(const uint8_t* a8_, int code_size) { set(a8_, code_size); }

    void set(const uint8_t* a8_2, int code_size) {
        a8 = a8_2;
        quotient8 = code_size / 8;
        remainder8 = code_size % 8;
        // compute pa ONCE here (uses a8, not a64)
        pa_cached = total_popcnt_bytes_avx512(a8, code_size);
    }

    // Fast Hamming using XOR+popcnt; AVX-512 path if available
    int hamming(const uint8_t* b8) const {
        int accu = 0;
        const uint64_t* a64 = reinterpret_cast<const uint64_t*>(a8);
        const uint64_t* b64 = reinterpret_cast<const uint64_t*>(b8);
        
        int i = 0;
        #if defined(__AVX512VPOPCNTDQ__)
            int blocks = quotient8 / 8; // 512-bit blocks

            for (; i < blocks; ++i) {
                __m512i va = _mm512_loadu_si512((const void*)&a64[i*8]);
                __m512i vb = _mm512_loadu_si512((const void*)&b64[i*8]);
                __m512i vx = _mm512_xor_si512(va, vb);
                __m512i pc = _mm512_popcnt_epi64(vx);
                alignas(64) uint64_t tmp[8];
                _mm512_store_si512((__m512i*)tmp, pc);
                accu += int(tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            }
            i *= 8; // words consumed
        #endif

        // scalar tail (Duff style)
        int len = quotient8 - i;
        switch (len & 7) {
        default:
            while (len > 7) {
                len -= 8;
                accu += popcount64(a64[i] ^ b64[i]); ++i;
                [[fallthrough]];
            case 7: accu += popcount64(a64[i] ^ b64[i]); ++i; [[fallthrough]];
            case 6: accu += popcount64(a64[i] ^ b64[i]); ++i; [[fallthrough]];
            case 5: accu += popcount64(a64[i] ^ b64[i]); ++i; [[fallthrough]];
            case 4: accu += popcount64(a64[i] ^ b64[i]); ++i; [[fallthrough]];
            case 3: accu += popcount64(a64[i] ^ b64[i]); ++i; [[fallthrough]];
            case 2: accu += popcount64(a64[i] ^ b64[i]); ++i; [[fallthrough]];
            case 1: accu += popcount64(a64[i] ^ b64[i]); ++i;
            }
        }
        // byte tail (rare for 4096-bit)
        if (remainder8) {
            const uint8_t* a = a8 + 8 * quotient8;
            const uint8_t* b = b8 + 8 * quotient8;
            static const uint8_t lut[256] = {/* same 256-entry popcnt table as above */};
            switch (remainder8) {
            case 7: accu += lut[a[6] ^ b[6]]; [[fallthrough]];
            case 6: accu += lut[a[5] ^ b[5]]; [[fallthrough]];
            case 5: accu += lut[a[4] ^ b[4]]; [[fallthrough]];
            case 4: accu += lut[a[3] ^ b[3]]; [[fallthrough]];
            case 3: accu += lut[a[2] ^ b[2]]; [[fallthrough]];
            case 2: accu += lut[a[1] ^ b[1]]; [[fallthrough]];
            case 1: accu += lut[a[0] ^ b[0]]; [[fallthrough]];
            default: break;
            }
        }
        return accu; // px (Hamming)
    }

    // Optional: Jaccard using cached pa and a precomputed/stored pb
    float jaccard_with_pbxx(const uint8_t* b8, int pb) const {
        int px = hamming(b8);
        int num = (pa_cached + pb - px);
        int den = (pa_cached + pb + px);
        return den ? float(num) / float(den) : 1.0f;
    }


    // ---------- Jaccard (fast path: requires precomputed pb per doc) ----------
    float jaccard_with_pb(const uint8_t* b8, int pb) const {
        const int px = hamming(b8);
        const int num = (pa_cached + pb - px);
        const int den = (pa_cached + pb + px);
        return den ? float(num) / float(den) : 1.0f;
    }

    // ---------- Jaccard (on-the-fly: compute pb + px together) ----------
    float jaccard_onthefly(const uint8_t* b8) const {
        const uint64_t* a64 = reinterpret_cast<const uint64_t*>(a8);
        const uint64_t* b64 = reinterpret_cast<const uint64_t*>(b8);
        int i = 0, pb = 0, px = 0;

    #if defined(__AVX512VPOPCNTDQ__)
        int blocks = quotient8 / 8;
        for (; i < blocks; ++i) {
            __m512i va = _mm512_loadu_si512((const void*)&a64[i*8]);
            __m512i vb = _mm512_loadu_si512((const void*)&b64[i*8]);
            __m512i vx = _mm512_xor_si512(va, vb);

            __m512i pcb = _mm512_popcnt_epi64(vb);
            __m512i pcx = _mm512_popcnt_epi64(vx);

            pb += hsum8_u64(pcb);
            px += hsum8_u64(pcx);
        }
        i *= 8;
    #endif
        int len = quotient8 - i;
        switch (len & 7) {
        default:
            while (len > 7) {
                len -= 8;
                pb += popcnt64_scalar(b64[i]);
                px += popcnt64_scalar(a64[i] ^ b64[i]);
                ++i;
                [[fallthrough]];
            case 7: pb += popcount64(b64[i]); px += popcount64(a64[i] ^ b64[i]); ++i; [[fallthrough]];
            case 6: pb += popcount64(b64[i]); px += popcount64(a64[i] ^ b64[i]); ++i; [[fallthrough]];
            case 5: pb += popcount64(b64[i]); px += popcount64(a64[i] ^ b64[i]); ++i; [[fallthrough]];
            case 4: pb += popcount64(b64[i]); px += popcount64(a64[i] ^ b64[i]); ++i; [[fallthrough]];
            case 3: pb += popcount64(b64[i]); px += popcount64(a64[i] ^ b64[i]); ++i; [[fallthrough]];
            case 2: pb += popcount64(b64[i]); px += popcount64(a64[i] ^ b64[i]); ++i; [[fallthrough]];
            case 1: pb += popcount64(b64[i]); px += popcount64(a64[i] ^ b64[i]); ++i;
            }
        }
        if (remainder8) {
            extern const uint8_t hamdis_tab_ham_bytes[256];
            const uint8_t* a = a8 + 8 * quotient8;
            const uint8_t* b = b8 + 8 * quotient8;
            switch (remainder8) {
            case 7: pb += hamdis_tab_ham_bytes[b[6]]; px += hamdis_tab_ham_bytes[a[6] ^ b[6]]; [[fallthrough]];
            case 6: pb += hamdis_tab_ham_bytes[b[5]]; px += hamdis_tab_ham_bytes[a[5] ^ b[5]]; [[fallthrough]];
            case 5: pb += hamdis_tab_ham_bytes[b[4]]; px += hamdis_tab_ham_bytes[a[4] ^ b[4]]; [[fallthrough]];
            case 4: pb += hamdis_tab_ham_bytes[b[3]]; px += hamdis_tab_ham_bytes[a[3] ^ b[3]]; [[fallthrough]];
            case 3: pb += hamdis_tab_ham_bytes[b[2]]; px += hamdis_tab_ham_bytes[a[2] ^ b[2]]; [[fallthrough]];
            case 2: pb += hamdis_tab_ham_bytes[b[1]]; px += hamdis_tab_ham_bytes[a[1] ^ b[1]]; [[fallthrough]];
            case 1: pb += hamdis_tab_ham_bytes[b[0]]; px += hamdis_tab_ham_bytes[a[0] ^ b[0]]; [[fallthrough]];
            default: break;
            }
        }
        const int num = (pa_cached + pb - px);
        const int den = (pa_cached + pb + px);
        return den ? float(num) / float(den) : 1.0f;
    }

    inline int get_code_size() const { return quotient8 * 8 + remainder8; }
};


/***************************************************************************
 * generalized Hamming = number of bytes that are different between
 * two codes.
 ***************************************************************************/

inline int generalized_hamming_64(uint64_t a) {
    a |= a >> 1;
    a |= a >> 2;
    a |= a >> 4;
    a &= 0x0101010101010101UL;
    return popcount64(a);
}

struct GenHammingComputer8 {
    uint64_t a0;

    GenHammingComputer8(const uint8_t* a, int code_size) {
        assert(code_size == 8);
        a0 = *(uint64_t*)a;
    }

    inline int hamming(const uint8_t* b) const {
        return generalized_hamming_64(*(uint64_t*)b ^ a0);
    }

    inline static constexpr int get_code_size() {
        return 8;
    }
};

// I'm not sure whether this version is faster of slower, tbh
// todo: test on different CPUs
struct GenHammingComputer16 {
    __m128i a;

    GenHammingComputer16(const uint8_t* a8, int code_size) {
        assert(code_size == 16);
        a = _mm_loadu_si128((const __m128i_u*)a8);
    }

    inline int hamming(const uint8_t* b8) const {
        const __m128i b = _mm_loadu_si128((const __m128i_u*)b8);
        const __m128i cmp = _mm_cmpeq_epi8(a, b);
        const auto movemask = _mm_movemask_epi8(cmp);
        return 16 - popcount32(movemask);
    }

    inline static constexpr int get_code_size() {
        return 16;
    }
};

struct GenHammingComputer32 {
    __m256i a;

    GenHammingComputer32(const uint8_t* a8, int code_size) {
        assert(code_size == 32);
        a = _mm256_loadu_si256((const __m256i_u*)a8);
    }

    inline int hamming(const uint8_t* b8) const {
        const __m256i b = _mm256_loadu_si256((const __m256i_u*)b8);
        const __m256i cmp = _mm256_cmpeq_epi8(a, b);
        const uint32_t movemask = _mm256_movemask_epi8(cmp);
        return 32 - popcount32(movemask);
    }

    inline static constexpr int get_code_size() {
        return 32;
    }
};

// A specialized version might be needed for the very long
// GenHamming code_size. In such a case, one may accumulate
// counts using _mm256_sub_epi8 and then compute a horizontal
// sum (using _mm256_sad_epu8, maybe, in blocks of no larger
// than 256 * 32 bytes).

struct GenHammingComputerM8 {
    const uint64_t* a;
    int n;

    GenHammingComputerM8(const uint8_t* a8, int code_size) {
        assert(code_size % 8 == 0);
        a = (uint64_t*)a8;
        n = code_size / 8;
    }

    int hamming(const uint8_t* b8) const {
        const uint64_t* b = (uint64_t*)b8;
        int accu = 0;

        int i = 0;
        int n4 = (n / 4) * 4;
        for (; i < n4; i += 4) {
            const __m256i av = _mm256_loadu_si256((const __m256i_u*)(a + i));
            const __m256i bv = _mm256_loadu_si256((const __m256i_u*)(b + i));
            const __m256i cmp = _mm256_cmpeq_epi8(av, bv);
            const uint32_t movemask = _mm256_movemask_epi8(cmp);
            accu += 32 - popcount32(movemask);
        }

        for (; i < n; i++)
            accu += generalized_hamming_64(a[i] ^ b[i]);
        return accu;
    }

    inline int get_code_size() const {
        return n * 8;
    }
};

} // namespace faiss

#endif