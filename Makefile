COPTS  = -O2

all: matmul conv1d

matmul: matmul.c
	$(CC) $(COPTS) -I ggml/include -L ggml/build/src -o matmul matmul.c -l ggml-base -l ggml-cpu -l ggml

conv1d: conv1d.c
	$(CC) $(COPTS) -I ggml/include -L ggml/build/src -o conv1d conv1d.c -l ggml-base -l ggml-cpu -l ggml

clean:
	rm -f matmul
	rm -f conv1d

.PHONY: all clean
