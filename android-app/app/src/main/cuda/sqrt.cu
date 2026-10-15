// Element-wise square root, compiled to a 32-bit sm_32 cubin for Tegra K1 by nvcc-android (CUDA 7.5).
extern "C" __global__ void sqrt_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = sqrtf(in[i]);
}
