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
     * {1, 2, 3, 4, 5} using a kernel tensor {3, 4, 5}.
     */
    struct ggml_tensor *tA = ggml_arange(ctx, 1, 6, 1);

    struct ggml_tensor *tK_f32 = ggml_arange(ctx, 3, 6, 1);
    struct ggml_tensor *tK = ggml_cast(ctx, tK_f32, GGML_TYPE_F16);

    struct ggml_tensor *tR = ggml_conv_1d(ctx, tK, tA, 1, 1, 1);
    ggml_build_forward_expand(gf, tR);

    ggml_graph_compute_with_ctx(ctx, gf, 1);

    /*
     * The result must be {14, 26, 38, 50, 32}.
     */
    printf("[");
    for (int i = 0; i < ggml_nelements(tR); i++) {
        printf(" %-5.1f", ((float *) tR->data)[i]);
    }
    printf("]\n");

    ggml_free(ctx);
}
