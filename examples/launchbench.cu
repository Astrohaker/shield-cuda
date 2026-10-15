// x[i] += 1, a deliberately tiny kernel: launch overhead dominates.
extern "C" __global__ void inc(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
}
