# ggml-python-dev

## 2025-07-18 Interface Memo

### `struct ggml_cgraph * ggml_new_graph(struct ggml_context * ctx)`

Create an empty computation graph.

**Parameters:**

* `ggml_context`

  * The context object

**Return Value:**

A new `ggml_cgraph` object.

### `struct ggml_context * ggml_init (struct ggml_init_params params)`

Create a new `ggml_context` object.

A context object manages a memory buffer used for the tensor computation.
You can preallocate a memory buffer and pass it to this function call, or
let ggml to manage it internally.

**Parameters:**

* `ggml_init_params`

  * `mem_size <size_t>` : The size of the memory buffer in bytes
  * `mem_buffer <void *>`: A pre-allocated memory buffer. If NULL,
     ggml will allocate a memory pool internally.
  * `no_alloc <bool>`: Set true to avoid allocating tensor data in
     context's memory pool.

**Return Value:**

A `ggml_context` object

_Note:_ If the function fails to allocate the context struct, it calls abort(3).

## 2025-07-16 Big Picture

- Make GGML (essentially a fast tensor library for edge devices) easier
  to use from Python.

- Fiddle around GGML and have a bit of fun.
