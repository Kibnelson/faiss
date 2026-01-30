/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexBinaryHNSW.h>

#include <omp.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include <cstdint>

#include <faiss/IndexBinaryFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/random.h>

#include <xmmintrin.h>
#include <algorithm>
#include <random>


#include <omp.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include <cstdint>


namespace faiss {

/**************************************************************
 * add / search blocks of descriptors
 **************************************************************/


 // --- file-scope, above namespace or inside namespace faiss ---
static inline float fast_rcp(float x) {
#if defined(__SSE__)
    __m128 vx = _mm_set_ss(x);
    __m128 r  = _mm_rcp_ss(vx);                                   // ~12-bit
    r = _mm_mul_ss(r, _mm_sub_ss(_mm_set_ss(2.0f), _mm_mul_ss(vx, r))); // ~22-bit
    return _mm_cvtss_f32(r);
#else
    return 1.0f / x;
#endif
}


    std::mutex g_pb_mutex;  // one per translation unit

namespace {

void hnsw_add_vertices(
        IndexBinaryHNSW& index_hnsw,
        size_t n0,
        size_t n,
        const uint8_t* x,
        bool verbose,
        bool preset_levels = false) {
    HNSW& hnsw = index_hnsw.hnsw;
    size_t ntotal = n0 + n;
    double t0 = getmillisecs();
    if (verbose) {
        printf("hnsw_add_vertices: adding %zd elements on top of %zd "
               "(preset_levels=%d)\n",
               n,
               n0,
               int(preset_levels));
    }

    int max_level = hnsw.prepare_level_tab(n, preset_levels);

    if (verbose) {
        printf("  max_level = %d\n", max_level);
    }

    std::vector<omp_lock_t> locks(ntotal);
    for (int i = 0; i < ntotal; i++) {
        omp_init_lock(&locks[i]);
    }

    // add vectors from highest to lowest level
    std::vector<int> hist;
    std::vector<int> order(n);

    { // make buckets with vectors of the same level

        // build histogram
        for (int i = 0; i < n; i++) {
            HNSW::storage_idx_t pt_id = i + n0;
            int pt_level = hnsw.levels[pt_id] - 1;
            while (pt_level >= hist.size()) {
                hist.push_back(0);
            }
            hist[pt_level]++;
        }

        // accumulate
        std::vector<int> offsets(hist.size() + 1, 0);
        for (int i = 0; i < hist.size() - 1; i++) {
            offsets[i + 1] = offsets[i] + hist[i];
        }

        // bucket sort
        for (int i = 0; i < n; i++) {
            HNSW::storage_idx_t pt_id = i + n0;
            int pt_level = hnsw.levels[pt_id] - 1;
            order[offsets[pt_level]++] = pt_id;
        }
    }

    { // perform add
        RandomGenerator rng2(789);

        int i1 = n;

        for (int pt_level = hist.size() - 1;
             pt_level >= int(!index_hnsw.init_level0);
             pt_level--) {
            int i0 = i1 - hist[pt_level];

            if (verbose) {
                printf("Adding %d elements at level %d\n", i1 - i0, pt_level);
            }

            // random permutation to get rid of dataset order bias
            for (int j = i0; j < i1; j++) {
                std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);
            }

#pragma omp parallel
            {
                VisitedTable vt(ntotal);

                std::unique_ptr<DistanceComputer> dis(
                        index_hnsw.get_distance_computer());
                int prev_display =
                        verbose && omp_get_thread_num() == 0 ? 0 : -1;

#pragma omp for schedule(dynamic)
                for (int i = i0; i < i1; i++) {
                    HNSW::storage_idx_t pt_id = order[i];
                    dis->set_query(
                            (float*)(x + (pt_id - n0) * index_hnsw.code_size));

                    hnsw.add_with_locks(
                            *dis,
                            pt_level,
                            pt_id,
                            locks,
                            vt,
                            index_hnsw.keep_max_size_level0 && (pt_level == 0));

                    if (prev_display >= 0 && i - i0 > prev_display + 10000) {
                        prev_display = i - i0;
                        printf("  %d / %d\r", i - i0, i1 - i0);
                        fflush(stdout);
                    }
                }
            }
            i1 = i0;
        }
        if (index_hnsw.init_level0) {
            FAISS_ASSERT(i1 == 0);
        } else {
            FAISS_ASSERT((i1 - hist[0]) == 0);
        }
    }
    if (verbose) {
        printf("Done in %.3f ms\n", getmillisecs() - t0);
    }

    for (int i = 0; i < ntotal; i++) {
        omp_destroy_lock(&locks[i]);
    }
}

} // anonymous namespace

/**************************************************************
 * IndexBinaryHNSW implementation
 **************************************************************/

IndexBinaryHNSW::IndexBinaryHNSW() {
    is_trained = true;
}

IndexBinaryHNSW::IndexBinaryHNSW(int d, int M)
        : IndexBinary(d),
          hnsw(M),
          own_fields(true),
          storage(new IndexBinaryFlat(d)) {
    is_trained = true;
}

IndexBinaryHNSW::IndexBinaryHNSW(IndexBinary* storage, int M)
        : IndexBinary(storage->d),
          hnsw(M),
          own_fields(false),
          storage(storage) {
    is_trained = true;
}

IndexBinaryHNSW::~IndexBinaryHNSW() {
    if (own_fields) {
        delete storage;
    }
}

void IndexBinaryHNSW::train(idx_t n, const uint8_t* x) {
    // hnsw structure does not require training
    storage->train(n, x);
    is_trained = true;
}

void IndexBinaryHNSW::train(idx_t n, const void* x, NumericType numeric_type) {
    IndexBinary::train(n, x, numeric_type);
}

void IndexBinaryHNSW::search(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");
    FAISS_THROW_IF_NOT(k > 0);

    // we use the buffer for distances as float but convert them back
    // to int in the end
    float* distances_f = (float*)distances;

    using RH = HeapBlockResultHandler<HNSW::C>;
    RH bres(n, distances_f, labels, k);

#pragma omp parallel
    {
        VisitedTable vt(ntotal);
        std::unique_ptr<DistanceComputer> dis(get_distance_computer());
        RH::SingleResultHandler res(bres);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            res.begin(i);
            dis->set_query((float*)(x + i * code_size));
            hnsw.search(*dis, res, vt);
            res.end();
        }
    }

// #pragma omp parallel for
//     for (int i = 0; i < n * k; ++i) {
//         distances[i] = std::round(distances_f[i]);
//     }

    #pragma omp parallel for
    for (int i = 0; i < n * k; ++i) {
        // Jaccard distance in [0,1] → scale to [0,100] with 2 decimal places
        // distances[i] = (int32_t)std::lround(distances_f[i] * 100.0f);
        distances[i] = (int32_t)std::floor(distances_f[i] * 100.0f + 1e-6f);
    }
}

void IndexBinaryHNSW::search(
        idx_t n,
        const void* x,
        NumericType numeric_type,
        idx_t k,
        int32_t* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    IndexBinary::search(n, x, numeric_type, k, distances, labels, params);
}


void IndexBinaryHNSW::add(idx_t n, const uint8_t* x) {
    FAISS_THROW_IF_NOT(is_trained);
    int n0 = ntotal;
    storage->add(n, x);
    ntotal = storage->ntotal;
    
    ensure_pb_array_();  
    
    hnsw_add_vertices(*this, n0, n, x, verbose, hnsw.levels.size() == ntotal);
}


void IndexBinaryHNSW::add(idx_t n, const void* x, NumericType numeric_type) {
    IndexBinary::add(n, x, numeric_type);
}

void IndexBinaryHNSW::reset() {
    hnsw.reset();
    storage->reset();
    pb_array_.clear();   
    ntotal = 0;
}

void IndexBinaryHNSW::reconstruct(idx_t key, uint8_t* recons) const {
    storage->reconstruct(key, recons);
}


void IndexBinaryHNSW::ensure_pb_array_() const {
    auto* flat = dynamic_cast<const IndexBinaryFlat*>(storage);
    FAISS_ASSERT(flat);
    const idx_t new_nt = flat->ntotal;
    const int  cs      = flat->code_size;

    // Cheap fast-path under the mutex too (to serialize first-time build)
    std::lock_guard<std::mutex> lock(g_pb_mutex);

    const idx_t old_nt = pb_array_.size();
    if (new_nt <= old_nt) {
        return;
    }

    FAISS_ASSERT(cs % 8 == 0);
    const int n_words = cs / 8;

    const uint8_t* xb = flat->xb.data();
    FAISS_ASSERT((idx_t)flat->xb.size() == new_nt * (idx_t)cs);

    pb_array_.reserve(new_nt);
    pb_array_.resize(new_nt);

    // Now we can parallel-fill safely; no other thread can observe
    // "size == new_nt" until we're done with this function.
#pragma omp parallel for if (new_nt - old_nt >= 16384) schedule(static)
    for (idx_t i = old_nt; i < new_nt; ++i) {
        const uint64_t* w =
            reinterpret_cast<const uint64_t*>(xb + i * cs);
        int s = 0;

    #if defined(__AVX512VPOPCNTDQ__)
        if (n_words >= 8) {
            const int vec_limit = (n_words / 8) * 8;
            for (int j = 0; j < vec_limit; j += 8) {
                __m512i v  = _mm512_loadu_si512(
                        reinterpret_cast<const __m512i*>(&w[j]));
                __m512i pc = _mm512_popcnt_epi64(v);
                s += (int)_mm512_reduce_add_epi64(pc);
            }
            for (int j = vec_limit; j < n_words; ++j) {
                s += __builtin_popcountll(w[j]);
            }
        } else
    #endif
        {
            for (int j = 0; j < n_words; ++j) {
                s += __builtin_popcountll(w[j]);
            }
        }

        pb_array_[i] = (uint16_t)s;
    }
}


namespace {

template <class HammingComputer>
struct FlatHammingDis : DistanceComputer {
    const int code_size;
    const uint8_t* b;
    const uint16_t* pb;  
    size_t ndis;
    HammingComputer hc;

    // Example of how to use the non cached version  via the operator
    // float operator()(idx_t i) override {
    // ndis++;
    // float j = hc.jaccard_onthefly(b + i * code_size); // computes pb & px
    // return 1.0f - j;
    // }

    //  default one
float operator()(idx_t i) override {

    ndis++;
    const uint8_t* bi = b + size_t(i) * code_size;
    
    #if defined(__AVX512VPOPCNTDQ__)
        const uint64_t* a64 = (const uint64_t*)hc.a8;
        const uint64_t* b64 = (const uint64_t*)bi;
        
        __m512i acc = _mm512_setzero_si512();
        
        // Unroll and accumulate in vector register
        for (int blk = 0; blk < 8; ++blk) {
            __m512i va = _mm512_loadu_si512(&a64[blk * 8]);
            __m512i vb = _mm512_loadu_si512(&b64[blk * 8]);
            __m512i vx = _mm512_xor_si512(va, vb);
            acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(vx));
        }
        
        const int px = _mm512_reduce_add_epi64(acc);
        const int s = hc.pa_cached + pb[i];
        const int den_i = s + px;
        
    
        const float den = float(den_i + (den_i == 0));
        float v = (2.0f * float(px)) / den; 

        return v;

    #endif
    // Generic fallback
    const int px = hc.hamming(bi);
    const int s = hc.pa_cached + pb[i];
    const int den_i = s + px;
    const float den = float(den_i + (den_i == 0));
    float v = (2.0f * float(px)) / den; 
    return v;

}
// Example of how to use the non cached version  via the symmetric_dis
// float symmetric_dis(idx_t i, idx_t j) override {
// HammingComputerDefault hi(b + i * code_size, code_size);
// float jacc = hi.jaccard_onthefly(b + j * code_size);
// return 1.0f - jacc;
// }


float symmetric_dis(idx_t i, idx_t j) override {

    const uint64_t* xi = reinterpret_cast<const uint64_t*>(b + size_t(i) * code_size);
    const uint64_t* xj = reinterpret_cast<const uint64_t*>(b + size_t(j) * code_size);
    
    const int nwords = code_size / 8;
    int px = 0;
    
    #if defined(__AVX512VPOPCNTDQ__)
    if (nwords >= 8) {
        __m512i acc = _mm512_setzero_si512();
        const int vec_limit = (nwords / 8) * 8;
        
        for (int blk = 0; blk < vec_limit; blk += 8) {
            __m512i va = _mm512_loadu_si512(&xi[blk]);
            __m512i vb = _mm512_loadu_si512(&xj[blk]);
            __m512i vx = _mm512_xor_si512(va, vb);
            acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(vx));
        }
        
        px = _mm512_reduce_add_epi64(acc);
        
        // Scalar tail
        for (int k = vec_limit; k < nwords; ++k) {
            px += __builtin_popcountll(xi[k] ^ xj[k]);
        }
    } else {
        for (int k = 0; k < nwords; ++k) {
            px += __builtin_popcountll(xi[k] ^ xj[k]);
        }
    }
    #else
    for (int k = 0; k < nwords; ++k) {
        px += __builtin_popcountll(xi[k] ^ xj[k]);
    }
    #endif
    
    const int s = int(pb[i]) + int(pb[j]);
    const int den_i = s + px;
    const float den = float(den_i + (den_i == 0));
    return(2.0f * float(px)) / den; 
}


    explicit FlatHammingDis(const IndexBinaryFlat& storage,const uint16_t* pb_array)
            : code_size(storage.code_size),
              b(storage.xb.data()),
              pb(pb_array),
              ndis(0),
              hc() {}
            
    // void set_query(const float* x) override {
    //     hc.set((uint8_t*)x, code_size);
    // }

    // Consider prefetching in hot loops:
void set_query(const float* x) override {
    hc.set((uint8_t*)x, code_size);
    _mm_prefetch((const char*)b, _MM_HINT_T0);  // Prefetch first vector
}

    ~FlatHammingDis() override {
#pragma omp critical
        { hnsw_stats.ndis += ndis; }
    }
};


struct BuildDistanceComputer {
    using T = DistanceComputer*;
        const uint16_t* pb;                   // <-- hold pb pointer
        explicit BuildDistanceComputer(const uint16_t* pb_) : pb(pb_) {}
    template <class HammingComputer>
    DistanceComputer* f(IndexBinaryFlat* flat_storage) {
        return new FlatHammingDis<HammingComputer>(*flat_storage,pb);
    }
};

} // namespace



DistanceComputer* IndexBinaryHNSW::get_distance_computer() const {
    auto* flat = dynamic_cast<IndexBinaryFlat*>(storage);
    FAISS_ASSERT(flat);
    // Only compute if array is stale (works for both add and load scenarios)
    if (pb_array_.size() < (size_t)flat->ntotal) {
        ensure_pb_array_();  // Builds entire array from scratch on first call
    }

    FAISS_ASSERT((idx_t)pb_array_.size() >= flat->ntotal); // pb ready


    BuildDistanceComputer bd(pb_array_.data());
    return dispatch_HammingComputer(code_size, bd, flat);
}

/**************************************************************
 * IndexBinaryHNSWCagra implementation
 **************************************************************/

IndexBinaryHNSWCagra::IndexBinaryHNSWCagra() : IndexBinaryHNSW() {
    storage = nullptr;
}

IndexBinaryHNSWCagra::IndexBinaryHNSWCagra(int d, int M)
        : IndexBinaryHNSW(d, M) {
    init_level0 = true;
    keep_max_size_level0 = true;
}

void IndexBinaryHNSWCagra::add(idx_t n, const uint8_t* x) {
    FAISS_THROW_IF_NOT_MSG(
            !base_level_only,
            "Cannot add vectors when base_level_only is set to True");

    IndexBinaryHNSW::add(n, x);
}

void IndexBinaryHNSWCagra::search(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    if (!base_level_only) {
        IndexBinaryHNSW::search(n, x, k, distances, labels, params);
    } else {
        float* distances_f = (float*)distances;

        using RH = HeapBlockResultHandler<HNSW::C>;
        RH bres(n, distances_f, labels, k);

        std::vector<storage_idx_t> nearest(n);
        std::vector<float> nearest_d(n);

#pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            std::unique_ptr<DistanceComputer> dis(get_distance_computer());
            dis->set_query((float*)(x + i * code_size));

            nearest[i] = -1;
            nearest_d[i] = std::numeric_limits<float>::max();

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<idx_t> distrib(0, this->ntotal - 1);

            for (idx_t j = 0; j < num_base_level_search_entrypoints; j++) {
                auto idx = distrib(gen);
                float distance = (*dis)(idx);

                if (distance < nearest_d[i]) {
                    nearest[i] = idx;
                    nearest_d[i] = distance;
                }
            }
            FAISS_THROW_IF_NOT_MSG(
                    nearest[i] >= 0, "Could not find a valid entrypoint.");
        }

#pragma omp parallel
        {
            VisitedTable vt(ntotal);
            std::unique_ptr<DistanceComputer> dis(get_distance_computer());
            HNSWStats search_stats;
            RH::SingleResultHandler res(bres);

#pragma omp for
            for (idx_t i = 0; i < n; i++) {
                res.begin(i);
                dis->set_query((float*)(x + i * code_size));

                hnsw.search_level_0(
                        *dis,
                        res,
                        1,
                        &nearest[i],
                        &nearest_d[i],
                        1, // search_type
                        search_stats,
                        vt,
                        params);

                res.end();
            }
        }

#pragma omp parallel for
        for (int i = 0; i < n * k; ++i) {
            distances[i] = std::round(distances_f[i]);
        }
    }
}

} // namespace faiss