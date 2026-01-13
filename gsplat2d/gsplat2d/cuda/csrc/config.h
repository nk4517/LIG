#define MAX_BLOCK_SIZE ( 16 * 16 )
#define N_THREADS 256

#define MAX_REGISTER_CHANNELS 3

#define CUDA_CALL(x)                                                           \
    do {                                                                       \
        if ((x) != cudaSuccess) {                                              \
            printf(                                                            \
                "Error at %s:%d - %s\n",                                       \
                __FILE__,                                                      \
                __LINE__,                                                      \
                cudaGetErrorString(cudaGetLastError())                         \
            );                                                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

enum RasterizeExtras : unsigned {
    RASTERIZE_EXTRAS_NONE = 0,
    RASTERIZE_EXTRAS_T = 1 << 0,           // compute transmittance T
    RASTERIZE_EXTRAS_UPSCALE_GRADS = 1 << 1  // compute derivatives for upscale
};
