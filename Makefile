# Load CUDA using the following command
# module load cuda
#
CC = nvcc
CFLAGS = -O3 -arch=compute_35 -code=sm_35
NVCCFLAGS = -O3 -arch=compute_35 -code=sm_35
LIBS = 

TARGETS = matrix_cuda rk4_parallel

all:    $(TARGETS)

matrix_cuda: matrix_cuda.o
	$(CC) -o $@ $(NVCCLIBS) matrix_cuda.o

matrix_cuda.o: matrix_cuda.cu 
	$(CC) -c $(NVCCFLAGS) matrix_cuda.cu

rk4_parallel: rk4_parallel.o
	$(CC) -o $@ $(NVCCLIBS) rk4_parallel.o

rk4_parallel.o: rk4_parallel.cu
	$(CC) -c $(NVCCFLAGS) rk4_parallel.cu

clean:
	rm -f *.o $(TARGETS) *.stdout *.txt
