COPTS  = -O2

all: matmul conv1d conv1d_backend

matmul: matmul.c
	$(CC) $(COPTS) -I ggml/include -L ggml/build/src -o matmul matmul.c -l ggml-base -l ggml-cpu -l ggml

conv1d: conv1d.c
	$(CC) $(COPTS) -I ggml/include -L ggml/build/src -o conv1d conv1d.c -l ggml-base -l ggml-cpu -l ggml

conv1d_backend: conv1d_backend.c
	$(CC) $(COPTS) -I ggml/include -L ggml/build/src -o conv1d_backend conv1d_backend.c -l ggml-base -l ggml-cpu -l ggml

clean:
	rm -f matmul
	rm -f conv1d
	rm -f conv1d_backend

.PHONY: all clean
