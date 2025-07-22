#include "ggml.h"
#include "ggml-cpu.h"
#include <string.h>

/*
 * This function computes the exact amount needed for A * B^T
 * where A and B are both 2 x 2 tensors.
 */
size_t get_required_memory(void)
{
    size_t ret;
    ret += 2 * 2 * ggml_type_size(GGML_TYPE_F32) + ggml_tensor_overhead(); // A
    ret += 2 * 2 * ggml_type_size(GGML_TYPE_F32) + ggml_tensor_overhead(); // B
    ret += 2 * 2 * ggml_type_size(GGML_TYPE_F32) + ggml_tensor_overhead(); // A * B^T
    ret += ggml_graph_overhead();  // gf (computation graph)

    /* This is for a work memory buffer object. While matmul does not
     * require a work buffer, an empty buffer is allocated anyway.
     *
     * Note: `sizeof(struct ggml_object) == 32`
     */
    ret += 32;

    return ret;  // 83792 bytes
}

int main(int argc, char **argv)
{
    float A[4] = {1, 2, 3, 4};
    float B[4] = {4, 3, 2, 1};

    /*
     * Build Computation Graph
     */
    struct ggml_init_params params = {
        get_required_memory(),
        NULL,
        false,
    };

    struct ggml_context *ctx = ggml_init(params);

    struct ggml_tensor * tA = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    struct ggml_tensor * tB = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    memcpy(tA->data, A, ggml_nbytes(tA));
    memcpy(tB->data, B, ggml_nbytes(tB));

    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    struct ggml_tensor *ret = ggml_mul_mat(ctx, tA, tB);

    ggml_build_forward_expand(gf, ret);
    ggml_graph_compute_with_ctx(ctx, gf, 2); /* 2 threads */

    /*
     * Show Result
     */
    float *data = (float *) ret->data;
    for (int j = 0; j < ret->ne[1]; j++) {
        for (int i = 0; i < ret->ne[0]; i++) {
            printf("%5.1f", data[j * ret->ne[0] + i]);
        }
        printf("\n");
    }

    ggml_free(ctx);
    return 0;
}
