// k1torch GPU kernels for Tegra K1 (sm_32), built into kernels.cubin with nvcc-k1.
// Layouts: images NCHW, matrices row-major, all float32 except labels/indices (int32).

#define FOR_EACH(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// ---- elementwise ---------------------------------------------------------------------------

extern "C" __global__ void fill(float* x, float value, int n) {
    FOR_EACH(i, n) x[i] = value;
}

// y += a * x
extern "C" __global__ void axpy(float* y, const float* x, float a, int n) {
    FOR_EACH(i, n) y[i] += a * x[i];
}

// x *= s[0] (a scalar that lives on the GPU, so no round trip is needed)
extern "C" __global__ void scale_by(float* x, const float* s, int n) {
    FOR_EACH(i, n) x[i] *= s[0];
}

extern "C" __global__ void relu_forward(const float* x, float* y, int n) {
    FOR_EACH(i, n) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

extern "C" __global__ void relu_backward(const float* x, const float* dy, float* dx, int n) {
    FOR_EACH(i, n) dx[i] = x[i] > 0.0f ? dy[i] : 0.0f;
}

// dst[r] = src[idx[r]] for rows of `row` elements (batch shuffling on the GPU).
extern "C" __global__ void gather_rows(const float* src, const int* idx, float* dst, int rows, int row) {
    FOR_EACH(i, rows * row) dst[i] = src[idx[i / row] * row + i % row];
}

extern "C" __global__ void gather_int(const int* src, const int* idx, int* dst, int n) {
    FOR_EACH(i, n) dst[i] = src[idx[i]];
}

// ---- matrix multiply -----------------------------------------------------------------------
// C[M,N] = alpha * op(A)[M,K] @ op(B)[K,N] + beta * C. op = transpose when the flag is set:
// A is stored [M,K] (or [K,M] if transA), B is [K,N] (or [N,K] if transB).

#define TILE 16

extern "C" __global__ void gemm(const float* A, const float* B, float* C, int M, int N, int K,
                                int transA, int transB, float alpha, float beta) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE + 1];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        int ak = t + threadIdx.x;  // A[row, ak]
        int bk = t + threadIdx.y;  // B[bk, col]
        As[threadIdx.y][threadIdx.x] =
            row < M && ak < K ? (transA ? A[ak * M + row] : A[row * K + ak]) : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            bk < K && col < N ? (transB ? B[col * K + bk] : B[bk * N + col]) : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) {
        float* c = &C[row * N + col];
        *c = alpha * acc + (beta != 0.0f ? beta * *c : 0.0f);
    }
}

// y[n, f] += b[f]
extern "C" __global__ void add_bias_rows(float* y, const float* b, int rows, int features) {
    FOR_EACH(i, rows * features) y[i] += b[i % features];
}

// db[f] (+)= sum_n dy[n, f]
extern "C" __global__ void bias_grad_rows(const float* dy, float* db, int rows, int features,
                                          int accumulate) {
    FOR_EACH(f, features) {
        float s = 0.0f;
        for (int n = 0; n < rows; n++) s += dy[n * features + f];
        db[f] = accumulate ? db[f] + s : s;
    }
}

// ---- convolution (direct, NCHW, square kernel) ---------------------------------------------

extern "C" __global__ void conv2d_forward(const float* x, const float* w, const float* b, float* y,
                                          int N, int C, int H, int W, int OC, int K, int stride,
                                          int pad, int OH, int OW) {
    FOR_EACH(i, N * OC * OH * OW) {
        int ow = i % OW, oh = i / OW % OH, oc = i / (OW * OH) % OC, n = i / (OW * OH * OC);
        float acc = b ? b[oc] : 0.0f;
        for (int c = 0; c < C; c++) {
            const float* xc = x + (n * C + c) * H * W;
            const float* wc = w + (oc * C + c) * K * K;
            for (int kh = 0; kh < K; kh++) {
                int h = oh * stride - pad + kh;
                if (h < 0 || h >= H) continue;
                for (int kw = 0; kw < K; kw++) {
                    int ww = ow * stride - pad + kw;
                    if (ww >= 0 && ww < W) acc += xc[h * W + ww] * wc[kh * K + kw];
                }
            }
        }
        y[i] = acc;
    }
}

// dx[n,c,h,w] = sum over (oc, kh, kw) that read this pixel
extern "C" __global__ void conv2d_backward_input(const float* dy, const float* w, float* dx, int N,
                                                 int C, int H, int W, int OC, int K, int stride,
                                                 int pad, int OH, int OW) {
    FOR_EACH(i, N * C * H * W) {
        int ww = i % W, h = i / W % H, c = i / (W * H) % C, n = i / (W * H * C);
        float acc = 0.0f;
        for (int oc = 0; oc < OC; oc++) {
            const float* dyo = dy + (n * OC + oc) * OH * OW;
            const float* wo = w + (oc * C + c) * K * K;
            for (int kh = 0; kh < K; kh++) {
                int t = h + pad - kh;
                if (t < 0 || t % stride) continue;
                int oh = t / stride;
                if (oh >= OH) continue;
                for (int kw = 0; kw < K; kw++) {
                    int u = ww + pad - kw;
                    if (u < 0 || u % stride) continue;
                    int ow = u / stride;
                    if (ow < OW) acc += dyo[oh * OW + ow] * wo[kh * K + kw];
                }
            }
        }
        dx[i] = acc;
    }
}

// One block per weight element: reduce dy * x over (n, oh, ow). blockDim.x must be 128.
extern "C" __global__ void conv2d_backward_weight(const float* x, const float* dy, float* dw, int N,
                                                  int C, int H, int W, int OC, int K, int stride,
                                                  int pad, int OH, int OW, int accumulate) {
    __shared__ float part[128];
    int widx = blockIdx.x;  // (oc, c, kh, kw)
    int kw = widx % K, kh = widx / K % K, c = widx / (K * K) % C, oc = widx / (K * K * C);
    float acc = 0.0f;
    for (int j = threadIdx.x; j < N * OH * OW; j += blockDim.x) {
        int ow = j % OW, oh = j / OW % OH, n = j / (OW * OH);
        int h = oh * stride - pad + kh, ww = ow * stride - pad + kw;
        if (h >= 0 && h < H && ww >= 0 && ww < W) {
            acc += dy[((n * OC + oc) * OH + oh) * OW + ow] * x[((n * C + c) * H + h) * W + ww];
        }
    }
    part[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) part[threadIdx.x] += part[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) dw[widx] = accumulate ? dw[widx] + part[0] : part[0];
}

// One block per channel: db[oc] = sum dy[:, oc, :, :]. blockDim.x must be 128.
extern "C" __global__ void conv2d_backward_bias(const float* dy, float* db, int N, int OC, int HW,
                                                int accumulate) {
    __shared__ float part[128];
    int oc = blockIdx.x;
    float acc = 0.0f;
    for (int j = threadIdx.x; j < N * HW; j += blockDim.x) {
        acc += dy[(j / HW * OC + oc) * HW + j % HW];
    }
    part[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) part[threadIdx.x] += part[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) db[oc] = accumulate ? db[oc] + part[0] : part[0];
}

// ---- pooling -------------------------------------------------------------------------------

// Max pooling with window = stride = K; idx keeps the flat input index of each maximum.
extern "C" __global__ void maxpool_forward(const float* x, float* y, int* idx, int N, int C, int H,
                                           int W, int K, int OH, int OW) {
    FOR_EACH(i, N * C * OH * OW) {
        int ow = i % OW, oh = i / OW % OH, nc = i / (OW * OH);
        const float* xc = x + nc * H * W;
        int best = (oh * K) * W + ow * K;
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                int j = (oh * K + kh) * W + ow * K + kw;
                if (xc[j] > xc[best]) best = j;
            }
        }
        y[i] = xc[best];
        idx[i] = nc * H * W + best;
    }
}

// dx must be zeroed first; windows do not overlap, so plain stores are safe.
extern "C" __global__ void maxpool_backward(const float* dy, const int* idx, float* dx, int n) {
    FOR_EACH(i, n) dx[idx[i]] = dy[i];
}

// ---- loss and metrics ----------------------------------------------------------------------

// Softmax + cross-entropy per row. Writes grad = (softmax - onehot) / N and adds the mean loss
// to *loss (so losses can be accumulated on the GPU and read rarely).
extern "C" __global__ void cross_entropy(const float* logits, const int* labels, float* grad,
                                         float* loss, int N, int C) {
    FOR_EACH(n, N) {
        const float* z = logits + n * C;
        float m = z[0];
        for (int c = 1; c < C; c++) m = fmaxf(m, z[c]);
        float sum = 0.0f;
        for (int c = 0; c < C; c++) sum += __expf(z[c] - m);
        float logSum = __logf(sum) + m;
        int label = labels[n];
        for (int c = 0; c < C; c++) {
            float p = __expf(z[c] - logSum);
            grad[n * C + c] = (p - (c == label ? 1.0f : 0.0f)) / N;
        }
        atomicAdd(loss, (logSum - z[label]) / N);
    }
}

// *correct += number of rows whose argmax equals the label
extern "C" __global__ void count_correct(const float* logits, const int* labels, int* correct, int N,
                                         int C) {
    FOR_EACH(n, N) {
        const float* z = logits + n * C;
        int best = 0;
        for (int c = 1; c < C; c++) if (z[c] > z[best]) best = c;
        if (best == labels[n]) atomicAdd(correct, 1);
    }
}

// ---- optimizer -----------------------------------------------------------------------------

// SGD with momentum and weight decay: v = mu * v + g + wd * w; w -= lr * v
extern "C" __global__ void sgd_step(float* w, const float* g, float* v, float lr, float mu, float wd,
                                    int n) {
    FOR_EACH(i, n) {
        float d = g[i] + wd * w[i];
        float vi = mu * v[i] + d;
        v[i] = vi;
        w[i] -= lr * vi;
    }
}
