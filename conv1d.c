#include <ggml.h>
#include <ggml-cpu.h>
#include <string.h>
#include <inttypes.h>

int main(int argc, char **argv)
{
    struct ggml_init_params params = {
        1024 * 1024 * 1024, /* 1 MB */
        NULL,
        false
    };

    struct ggml_context *ctx = ggml_init(params);
    struct ggml_cgraph *gf = ggml_new_graph(ctx);

    /*
     * This applies an 1D convolution over an input tensor
     * {1, 2, 3, 4, 5} using a kernel tensor {10, 20, 10}.
     */
    struct ggml_tensor *tA = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 5, 1, 1);
    float *dA = (float *) tA->data;
    dA[0] = 1.0;
    dA[1] = 2.0;
    dA[2] = 3.0;
    dA[3] = 4.0;
    dA[4] = 5.0;

    struct ggml_tensor *tK = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 3, 1, 1);
    ggml_fp16_t *dK = (ggml_fp16_t *) tK->data;
    dK[0] = ggml_fp32_to_fp16(10.0);
    dK[1] = ggml_fp32_to_fp16(20.0);
    dK[2] = ggml_fp32_to_fp16(10.0);

    struct ggml_tensor *tR = ggml_conv_1d(ctx, tK, tA, 1, 1, 1);
    ggml_build_forward_expand(gf, tR);

    ggml_graph_compute_with_ctx(ctx, gf, 1);

    /*
     * The result must be {40, 80, 120, 160, 140}.
     */
    printf("[");
    for (int i = 0; i < ggml_nelements(tR); i++) {
        printf(" %-5.1f", ((float *) tR->data)[i]);
    }
    printf("]\n");

    ggml_free(ctx);
}
