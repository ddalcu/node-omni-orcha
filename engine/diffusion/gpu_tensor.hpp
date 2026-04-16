#ifndef __GPU_TENSOR_HPP__
#define __GPU_TENSOR_HPP__

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "tensor.hpp"
#include "rng.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <vector>

// A pool of same-shaped ggml tensors on a GPU backend, used for
// sampling-loop scratch data.  Tensors are acquired / released by
// GpuTensor's constructors and destructor so the pool never leaks.
struct SamplingGpuPool : std::enable_shared_from_this<SamplingGpuPool> {
    ggml_backend_t backend    = nullptr;
    ggml_context* pool_ctx    = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_gallocr_t graph_alloc = nullptr;

    std::vector<ggml_tensor*> available;
    int ndims = 0;
    int64_t ne[GGML_MAX_DIMS] = {};
    size_t tensor_bytes = 0;

    SamplingGpuPool(ggml_backend_t be, const std::vector<int64_t>& shape, int pool_size = 20)
        : backend(be) {
        ndims = std::min(static_cast<int>(shape.size()), GGML_MAX_DIMS);
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            ne[i] = (i < ndims) ? shape[i] : 1;
        }

        size_t overhead = ggml_tensor_overhead();
        ggml_init_params p = {};
        p.mem_size   = static_cast<size_t>(pool_size) * overhead + 256;
        p.mem_buffer = nullptr;
        p.no_alloc   = true;
        pool_ctx = ggml_init(p);

        available.reserve(pool_size);
        for (int i = 0; i < pool_size; i++) {
            ggml_tensor* t = ggml_new_tensor(pool_ctx, GGML_TYPE_F32, ndims, ne);
            available.push_back(t);
        }

        buf = ggml_backend_alloc_ctx_tensors(pool_ctx, backend);
        tensor_bytes = ggml_nbytes(available[0]);

        auto buf_type = ggml_backend_get_default_buffer_type(backend);
        graph_alloc = ggml_gallocr_new(buf_type);
    }

    ~SamplingGpuPool() {
        if (graph_alloc) ggml_gallocr_free(graph_alloc);
        if (buf) ggml_backend_buffer_free(buf);
        if (pool_ctx) ggml_free(pool_ctx);
    }

    SamplingGpuPool(const SamplingGpuPool&) = delete;
    SamplingGpuPool& operator=(const SamplingGpuPool&) = delete;

    ggml_tensor* acquire() {
        assert(!available.empty() && "SamplingGpuPool exhausted");
        ggml_tensor* t = available.back();
        available.pop_back();
        return t;
    }

    void release(ggml_tensor* t) {
        if (t) available.push_back(t);
    }

    int64_t numel() const {
        int64_t n = 1;
        for (int i = 0; i < ndims; i++) n *= ne[i];
        return n;
    }

    // Build a ggml graph rooted at `result_node`, execute it on the GPU
    // backend, then copy the output into `dst` (which must be a pool tensor
    // with an existing GPU buffer).
    void compute(ggml_tensor* dst, ggml_tensor* result_node, ggml_context* graph_ctx) {
        ggml_cgraph* gf = ggml_new_graph(graph_ctx);
        ggml_build_forward_expand(gf, result_node);
        ggml_gallocr_alloc_graph(graph_alloc, gf);
        if (ggml_backend_is_cpu(backend)) {
            ggml_backend_cpu_set_n_threads(backend, 4);
        }
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_tensor_copy(result_node, dst);
    }
};

// A GPU-resident tensor for use inside sampling loops.  Holds one tensor
// from a SamplingGpuPool and returns it on destruction.  Supports the same
// arithmetic operators as sd::Tensor<float> so that templated sampler
// functions work transparently.
class GpuTensor {
    ggml_tensor* tensor_ = nullptr;
    std::shared_ptr<SamplingGpuPool> pool_;

    // Helper: create a temporary ggml context large enough for a small graph
    static ggml_context* make_graph_ctx() {
        size_t mem = 16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false);
        ggml_init_params p = {};
        p.mem_size = mem;
        p.mem_buffer = nullptr;
        p.no_alloc = true;
        return ggml_init(p);
    }

    // Execute a unary op: dst = fn(a)
    static void exec_unary(SamplingGpuPool* pool, ggml_tensor* dst,
                           ggml_tensor* (*fn)(ggml_context*, ggml_tensor*),
                           ggml_tensor* a) {
        ggml_context* ctx = make_graph_ctx();
        ggml_tensor* r = fn(ctx, a);
        pool->compute(dst, r, ctx);
        ggml_free(ctx);
    }

    // Execute a binary op: dst = fn(a, b)
    static void exec_binary(SamplingGpuPool* pool, ggml_tensor* dst,
                            ggml_tensor* (*fn)(ggml_context*, ggml_tensor*, ggml_tensor*),
                            ggml_tensor* a, ggml_tensor* b) {
        ggml_context* ctx = make_graph_ctx();
        ggml_tensor* r = fn(ctx, a, b);
        pool->compute(dst, r, ctx);
        ggml_free(ctx);
    }

    // Create a scalar constant tensor on the GPU backend
    ggml_tensor* make_scalar_tensor(ggml_context* ctx, float val) const {
        ggml_tensor* s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        // gallocr will allocate for this; set_backend_tensor_data equivalent:
        // We'll handle differently — see scalar helper below.
        (void)val;
        return s;
    }

public:
    GpuTensor() = default;

    GpuTensor(ggml_tensor* t, std::shared_ptr<SamplingGpuPool> pool)
        : tensor_(t), pool_(std::move(pool)) {}

    ~GpuTensor() {
        if (tensor_ && pool_) pool_->release(tensor_);
    }

    // Move
    GpuTensor(GpuTensor&& o) noexcept : tensor_(o.tensor_), pool_(std::move(o.pool_)) {
        o.tensor_ = nullptr;
    }
    GpuTensor& operator=(GpuTensor&& o) noexcept {
        if (this != &o) {
            if (tensor_ && pool_) pool_->release(tensor_);
            tensor_ = o.tensor_;
            pool_ = std::move(o.pool_);
            o.tensor_ = nullptr;
        }
        return *this;
    }

    // Copy (GPU-to-GPU)
    GpuTensor(const GpuTensor& o) : pool_(o.pool_) {
        if (o.tensor_ && pool_) {
            tensor_ = pool_->acquire();
            ggml_backend_tensor_copy(o.tensor_, tensor_);
        }
    }
    GpuTensor& operator=(const GpuTensor& o) {
        if (this != &o) {
            if (!o.tensor_ || !o.pool_) {
                if (tensor_ && pool_) pool_->release(tensor_);
                tensor_ = nullptr;
                pool_ = nullptr;
            } else {
                pool_ = o.pool_;
                if (!tensor_) tensor_ = pool_->acquire();
                ggml_backend_tensor_copy(o.tensor_, tensor_);
            }
        }
        return *this;
    }

    // Convert from sd::Tensor<float> (CPU → GPU)
    GpuTensor(const sd::Tensor<float>& cpu, std::shared_ptr<SamplingGpuPool> p)
        : pool_(std::move(p)) {
        tensor_ = pool_->acquire();
        ggml_backend_tensor_set(tensor_, cpu.data(), 0, pool_->tensor_bytes);
    }

    // Convert to sd::Tensor<float> (GPU → CPU)
    sd::Tensor<float> to_cpu() const {
        if (!tensor_) return {};
        std::vector<int64_t> shape(pool_->ne, pool_->ne + pool_->ndims);
        sd::Tensor<float> result(shape);
        ggml_backend_tensor_get(tensor_, result.data(), 0, pool_->tensor_bytes);
        return result;
    }

    bool empty() const { return tensor_ == nullptr; }

    ggml_tensor* raw() const { return tensor_; }
    const std::shared_ptr<SamplingGpuPool>& get_pool() const { return pool_; }

    // Generate random tensor (CPU RNG for determinism, then upload)
    static GpuTensor randn_like(const GpuTensor& other, const std::shared_ptr<RNG>& rng) {
        if (!other.pool_) return {};
        auto pool = other.pool_;
        auto data = rng->randn(static_cast<uint32_t>(pool->numel()));
        ggml_tensor* t = pool->acquire();
        ggml_backend_tensor_set(t, data.data(), 0, pool->tensor_bytes);
        return GpuTensor(t, pool);
    }

    // ---------- Tensor-Tensor operators ----------

    friend GpuTensor operator+(const GpuTensor& a, const GpuTensor& b) {
        auto pool = a.pool_;
        ggml_tensor* dst = pool->acquire();
        exec_binary(pool.get(), dst, ggml_add, a.tensor_, b.tensor_);
        return GpuTensor(dst, pool);
    }

    friend GpuTensor operator-(const GpuTensor& a, const GpuTensor& b) {
        auto pool = a.pool_;
        ggml_tensor* dst = pool->acquire();
        exec_binary(pool.get(), dst, ggml_sub, a.tensor_, b.tensor_);
        return GpuTensor(dst, pool);
    }

    friend GpuTensor operator*(const GpuTensor& a, const GpuTensor& b) {
        auto pool = a.pool_;
        ggml_tensor* dst = pool->acquire();
        exec_binary(pool.get(), dst, ggml_mul, a.tensor_, b.tensor_);
        return GpuTensor(dst, pool);
    }

    friend GpuTensor operator/(const GpuTensor& a, const GpuTensor& b) {
        auto pool = a.pool_;
        ggml_tensor* dst = pool->acquire();
        exec_binary(pool.get(), dst, ggml_div, a.tensor_, b.tensor_);
        return GpuTensor(dst, pool);
    }

    // ---------- Tensor-Scalar operators ----------

    friend GpuTensor operator*(const GpuTensor& a, float s) {
        auto pool = a.pool_;
        ggml_tensor* dst = pool->acquire();
        ggml_context* ctx = make_graph_ctx();
        ggml_tensor* r = ggml_scale(ctx, a.tensor_, s);
        pool->compute(dst, r, ctx);
        ggml_free(ctx);
        return GpuTensor(dst, pool);
    }

    friend GpuTensor operator*(float s, const GpuTensor& a) {
        return a * s;
    }

    friend GpuTensor operator/(const GpuTensor& a, float s) {
        return a * (1.0f / s);
    }

    friend GpuTensor operator+(const GpuTensor& a, float s) {
        auto pool = a.pool_;
        ggml_tensor* dst = pool->acquire();
        ggml_context* ctx = make_graph_ctx();
        ggml_tensor* scalar = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, pool->numel());
        ggml_tensor* r = ggml_add(ctx, a.tensor_, scalar);

        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, r);
        ggml_gallocr_alloc_graph(pool->graph_alloc, gf);

        // Fill scalar tensor with the constant value
        std::vector<float> fill(static_cast<size_t>(pool->numel()), s);
        ggml_backend_tensor_set(scalar, fill.data(), 0, pool->tensor_bytes);

        if (ggml_backend_is_cpu(pool->backend)) {
            ggml_backend_cpu_set_n_threads(pool->backend, 4);
        }
        ggml_backend_graph_compute(pool->backend, gf);
        ggml_backend_tensor_copy(r, dst);

        ggml_free(ctx);
        return GpuTensor(dst, pool);
    }

    friend GpuTensor operator+(float s, const GpuTensor& a) {
        return a + s;
    }

    friend GpuTensor operator-(const GpuTensor& a, float s) {
        return a + (-s);
    }

    friend GpuTensor operator-(float s, const GpuTensor& a) {
        // s - a = -(a - s) = -(a + (-s))
        auto pool = a.pool_;
        ggml_tensor* dst = pool->acquire();
        ggml_context* ctx = make_graph_ctx();
        // -a * 1 + s  →  scale(-1) then add s
        ggml_tensor* neg = ggml_scale(ctx, a.tensor_, -1.0f);
        ggml_tensor* scalar = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, pool->numel());
        ggml_tensor* r = ggml_add(ctx, neg, scalar);

        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, r);
        ggml_gallocr_alloc_graph(pool->graph_alloc, gf);

        std::vector<float> fill(static_cast<size_t>(pool->numel()), s);
        ggml_backend_tensor_set(scalar, fill.data(), 0, pool->tensor_bytes);

        if (ggml_backend_is_cpu(pool->backend)) {
            ggml_backend_cpu_set_n_threads(pool->backend, 4);
        }
        ggml_backend_graph_compute(pool->backend, gf);
        ggml_backend_tensor_copy(r, dst);

        ggml_free(ctx);
        return GpuTensor(dst, pool);
    }

    friend GpuTensor operator/(float s, const GpuTensor& a) {
        // s / a  — element-wise.  Build: scale(reciprocal(a), s)
        // ggml doesn't have a reciprocal op, so use a filled tensor / a
        auto pool = a.pool_;
        ggml_tensor* dst = pool->acquire();
        ggml_context* ctx = make_graph_ctx();
        ggml_tensor* scalar = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, pool->numel());
        ggml_tensor* r = ggml_div(ctx, scalar, a.tensor_);

        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, r);
        ggml_gallocr_alloc_graph(pool->graph_alloc, gf);

        std::vector<float> fill(static_cast<size_t>(pool->numel()), s);
        ggml_backend_tensor_set(scalar, fill.data(), 0, pool->tensor_bytes);

        if (ggml_backend_is_cpu(pool->backend)) {
            ggml_backend_cpu_set_n_threads(pool->backend, 4);
        }
        ggml_backend_graph_compute(pool->backend, gf);
        ggml_backend_tensor_copy(r, dst);

        ggml_free(ctx);
        return GpuTensor(dst, pool);
    }

    // Unary negation
    friend GpuTensor operator-(const GpuTensor& a) {
        return a * (-1.0f);
    }

    // ---------- Compound assignment ----------

    friend GpuTensor& operator+=(GpuTensor& a, const GpuTensor& b) {
        auto pool = a.pool_;
        ggml_tensor* dst = pool->acquire();
        exec_binary(pool.get(), dst, ggml_add, a.tensor_, b.tensor_);
        pool->release(a.tensor_);
        a.tensor_ = dst;
        return a;
    }

    friend GpuTensor& operator-=(GpuTensor& a, const GpuTensor& b) {
        auto pool = a.pool_;
        ggml_tensor* dst = pool->acquire();
        exec_binary(pool.get(), dst, ggml_sub, a.tensor_, b.tensor_);
        pool->release(a.tensor_);
        a.tensor_ = dst;
        return a;
    }

    friend GpuTensor& operator*=(GpuTensor& a, const GpuTensor& b) {
        auto pool = a.pool_;
        ggml_tensor* dst = pool->acquire();
        exec_binary(pool.get(), dst, ggml_mul, a.tensor_, b.tensor_);
        pool->release(a.tensor_);
        a.tensor_ = dst;
        return a;
    }

    friend GpuTensor& operator*=(GpuTensor& a, float s) {
        auto pool = a.pool_;
        ggml_tensor* dst = pool->acquire();
        ggml_context* ctx = make_graph_ctx();
        ggml_tensor* r = ggml_scale(ctx, a.tensor_, s);
        pool->compute(dst, r, ctx);
        ggml_free(ctx);
        pool->release(a.tensor_);
        a.tensor_ = dst;
        return a;
    }

    friend GpuTensor& operator+=(GpuTensor& a, float s) {
        a = a + s;
        return a;
    }

    friend GpuTensor& operator-=(GpuTensor& a, float s) {
        a = a + (-s);
        return a;
    }

    friend GpuTensor& operator/=(GpuTensor& a, float s) {
        a *= (1.0f / s);
        return a;
    }
};

#endif  // __GPU_TENSOR_HPP__
