COPTS  = -O2

matmul: matmul.c
	$(CC) $(COPTS) -I ggml/include -L ggml/build/src -o matmul matmul.c -l ggml-base -l ggml-cpu -l ggml

clean:
	rm -f matmul

.PHONY: clean
