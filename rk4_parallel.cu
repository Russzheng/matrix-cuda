#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <ctime>
#include "matrix_cuda.cuh"
using namespace std;

#define BLOCK_SIZE 16
double* WEIGHTS;
int ROW;
int COL;

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void init_mat(double* a, double minVal, double maxVal) {
	for (int i = 0; i < ROW * COL; i++) {
		a[i] = fRand(minVal, maxVal);
	}

	for (int i = 0; i < COL * COL; i++) {
		WEIGHTS[i] = fRand(minVal, maxVal);
	}
}

void cpu_matrix_mult(double *h_a, double *h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < k; ++j) 
        {
            double tmp = 0.0;
            for (int h = 0; h < n; ++h) 
            {
                tmp += h_a[i * n + h] * WEIGHTS[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

void Function(double t, double* z, double* f) {
	// Serial Matmul
	cpu_matrix_mult(z, f, ROW, COL, COL);
}

void rk4(double t_init, double t_stop, double* input_mat, double* output_mat, int iteration, 
	double* previous_f0, double* previous_f1, double* previous_f2, double* previous_f3) {

	float gpu_elapsed_time_ms;
	// some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start to count execution time of GPU version
    cudaEventRecord(start, 0);

	double dt = t_stop - t_init;

	// Init
	//
	// Parallel
	//
	// Init
	double *weights_cuda, *outputs_cuda;
	double *z0_cuda, *z1_cuda, *z2_cuda, *z3_cuda;
	double *f0_cuda, *f1_cuda, *f2_cuda, *f3_cuda;
	cudaMalloc((void **) &weights_cuda, sizeof(double)*COL*COL);
	cudaMalloc((void **) &z0_cuda, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &z1_cuda, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &z2_cuda, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &z3_cuda, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &f0_cuda, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &f1_cuda, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &f2_cuda, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &f3_cuda, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &outputs_cuda, sizeof(double)*ROW*COL);

	// Collecting stages
	// double *tmp1, *tmp2;
	// cudaMalloc((void **) &tmp1, sizeof(double)*ROW*COL);
	// cudaMalloc((void **) &tmp2, sizeof(double)*ROW*COL);

    cudaMemcpy(z0_cuda, input_mat, sizeof(double)*ROW*COL, cudaMemcpyHostToDevice);
    cudaMemcpy(weights_cuda, WEIGHTS, sizeof(double)*COL*COL, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (ROW + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (COL + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	/// Stage 0
	//Parallel
    if(ROW == COL)
    {           
        gpu_square_matrix_mult<<<dimGrid, dimBlock>>>(z0_cuda, weights_cuda, f0_cuda, COL);    
    }
    else
    {
        gpu_matrix_mult<<<dimGrid, dimBlock>>>(z0_cuda, weights_cuda, f0_cuda, ROW, COL, COL);    
    }


	/// Stage 1	
	// Function(t1, z1, f1);
    gpu_matrix_add<<<2, (ROW * COL + 1) / 2>>>(z0_cuda, f0_cuda, z1_cuda, ROW*COL, 0.5 * dt);
	if(ROW == COL)
    {           
        gpu_square_matrix_mult<<<dimGrid, dimBlock>>>(z1_cuda, weights_cuda, f1_cuda, COL);    
    }
    else
    {
        gpu_matrix_mult<<<dimGrid, dimBlock>>>(z1_cuda, weights_cuda, f1_cuda, ROW, COL, COL);    
    }

	/// Stage 2
	gpu_matrix_add<<<2, (ROW * COL + 1) / 2>>>(z0_cuda, f1_cuda, z2_cuda, ROW*COL, 0.5 * dt);
	if(ROW == COL)
    {           
        gpu_square_matrix_mult<<<dimGrid, dimBlock>>>(z2_cuda, weights_cuda, f2_cuda, COL);    
    }
    else
    {
        gpu_matrix_mult<<<dimGrid, dimBlock>>>(z2_cuda, weights_cuda, f2_cuda, ROW, COL, COL);    
    }


	/// Stage 3
	gpu_matrix_add<<<2, (ROW * COL + 1) / 2>>>(z0_cuda, f2_cuda, z3_cuda, ROW*COL, 0.5 * dt);
	if(ROW == COL)
    {           
        gpu_square_matrix_mult<<<dimGrid, dimBlock>>>(z3_cuda, weights_cuda, f3_cuda, COL);    
    }
    else
    {
        gpu_matrix_mult<<<dimGrid, dimBlock>>>(z3_cuda, weights_cuda, f3_cuda, ROW, COL, COL);    
    }

	/// Collect Stages:
	gpu_matrix_add_multi<<<2, (ROW * COL + 1) / 2>>>(z0_cuda, f0_cuda, f1_cuda, f2_cuda, f3_cuda, outputs_cuda, 
		ROW*COL, 1.0, dt / 6.0, dt / 3.0, dt / 3.0, dt / 6.0);

	/// save previous K:
	cudaMemcpy(previous_f0, f0_cuda, sizeof(double)*ROW*COL, cudaMemcpyDeviceToHost);
	cudaMemcpy(previous_f1, f1_cuda, sizeof(double)*ROW*COL, cudaMemcpyDeviceToHost);
	cudaMemcpy(previous_f2, f2_cuda, sizeof(double)*ROW*COL, cudaMemcpyDeviceToHost);
	cudaMemcpy(previous_f3, f3_cuda, sizeof(double)*ROW*COL, cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    // printf("Time elapsed on matrix multiplication of on GPU: %f ms.\n\n", gpu_elapsed_time_ms);

	cudaFree(f0_cuda);
	cudaFree(f1_cuda);
	cudaFree(z1_cuda);
	cudaFree(f2_cuda);
	cudaFree(z2_cuda);
	cudaFree(f3_cuda);
	cudaFree(z3_cuda);
	cudaFree(weights_cuda);
}

void parallelRK4(double t_init, double t_stop, double* input_mat, double* output_mat, int iteration, 
	double* previous_f0, double* previous_f1, double* previous_f2, double* previous_f3) {
	
	cudaStream_t stream0, stream1, stream2, stream3;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);

	double dt = t_stop - t_init;

	// Init
	//
	// Parallel
	//
	// Init
	double *weights_cuda, *outputs_cuda;
	double *z0_cuda, *z1_cuda, *z2_cuda, *z3_cuda;
	double *f0_cuda, *f1_cuda, *f2_cuda, *f3_cuda;
	double *pre_f0, *pre_f1, *pre_f2, *pre_f3;

	// Allocate memories
	cudaMalloc((void **) &weights_cuda, sizeof(double)*COL*COL);
	cudaMalloc((void **) &outputs_cuda, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &z0_cuda, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &f0_cuda, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &pre_f0, sizeof(double)*ROW*COL);

	cudaMalloc((void **) &z1_cuda, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &f1_cuda, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &pre_f1, sizeof(double)*ROW*COL);

	cudaMalloc((void **) &z2_cuda, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &f2_cuda, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &pre_f2, sizeof(double)*ROW*COL);

	cudaMalloc((void **) &z3_cuda, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &f3_cuda, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &pre_f3, sizeof(double)*ROW*COL);

	// Copy from Host to Device
	cudaMemcpy(z0_cuda, input_mat, sizeof(double)*ROW*COL, cudaMemcpyHostToDevice);
	cudaMemcpy(pre_f0, previous_f0, sizeof(double)*ROW*COL, cudaMemcpyHostToDevice);
	cudaMemcpy(weights_cuda, WEIGHTS, sizeof(double)*COL*COL, cudaMemcpyHostToDevice);

	cudaMemcpy(pre_f1, previous_f1, sizeof(double)*ROW*COL, cudaMemcpyHostToDevice);

	cudaMemcpy(pre_f2, previous_f2, sizeof(double)*ROW*COL, cudaMemcpyHostToDevice);

    cudaMemcpy(pre_f3, previous_f3, sizeof(double)*ROW*COL, cudaMemcpyHostToDevice);


    unsigned int grid_rows = (ROW + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (COL + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    float gpu_elapsed_time_ms;
	// some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start to count execution time of GPU version
    cudaEventRecord(start, 0);
    // printf("Here");

	// Parallel
	gpu_matrix_add_multi_four<<<2, (ROW * COL + 1) / 2, 0, stream1>>>(z0_cuda, pre_f0, pre_f1, pre_f3, z1_cuda, 
		ROW*COL, 1.0, dt * -0.75, dt * 0.5, dt * 0.75);

	gpu_matrix_add_multi_four<<<2, (ROW * COL + 1) / 2, 0, stream2>>>(z0_cuda, pre_f0, pre_f1, pre_f2, z2_cuda, 
		ROW*COL, 1.0, dt * -1, dt * 2.0, dt * -0.5);

	gpu_matrix_add_multi_three<<<2, (ROW * COL + 1) / 2, 0, stream3>>>(z0_cuda, pre_f0, pre_f3, z3_cuda, 
		ROW*COL, 1.0, dt * 0.5, dt * 0.5);	

	/// Stage 0
	//Parallel
    if(ROW == COL)
    {           
        gpu_square_matrix_mult<<<dimGrid, dimBlock, 0, stream0>>>(z0_cuda, weights_cuda, f0_cuda, COL);    
    }
    else
    {
        gpu_matrix_mult<<<dimGrid, dimBlock, 0, stream0>>>(z0_cuda, weights_cuda, f0_cuda, ROW, COL, COL);    
    }


	/// Stage 1	
	// Function(t1, z1, f1);
	if(ROW == COL)
    {           
        gpu_square_matrix_mult<<<dimGrid, dimBlock, 0, stream1>>>(z1_cuda, weights_cuda, f1_cuda, COL);    
    }
    else
    {
        gpu_matrix_mult<<<dimGrid, dimBlock, 0, stream1>>>(z1_cuda, weights_cuda, f1_cuda, ROW, COL, COL);    
    }

	/// Stage 2
	if(ROW == COL)
    {           
        gpu_square_matrix_mult<<<dimGrid, dimBlock, 0, stream2>>>(z2_cuda, weights_cuda, f2_cuda, COL);    
    }
    else
    {
        gpu_matrix_mult<<<dimGrid, dimBlock, 0, stream2>>>(z2_cuda, weights_cuda, f2_cuda, ROW, COL, COL);    
    }


	/// Stage 3
	if(ROW == COL)
    {           
        gpu_square_matrix_mult<<<dimGrid, dimBlock, 0, stream3>>>(z3_cuda, weights_cuda, f3_cuda, COL);    
    }
    else
    {
        gpu_matrix_mult<<<dimGrid, dimBlock, 0, stream3>>>(z3_cuda, weights_cuda, f3_cuda, ROW, COL, COL);    
    }

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
	cudaStreamSynchronize(stream3);
	/// Collect Stages:
	gpu_matrix_add_multi<<<2, (ROW * COL + 1) / 2, 0, stream0>>>(z0_cuda, f0_cuda, f1_cuda, f2_cuda, f3_cuda, outputs_cuda, 
			ROW*COL, 1.0, dt / 6.0, dt / 3.0, dt / 3.0, dt / 6.0);
	cudaStreamSynchronize(stream0);

	cudaThreadSynchronize();
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

	/// save previous K:

	cudaMemcpy(previous_f0, f0_cuda, sizeof(double)*ROW*COL, cudaMemcpyDeviceToHost);

	cudaMemcpy(previous_f1, f1_cuda, sizeof(double)*ROW*COL, cudaMemcpyDeviceToHost);

	cudaMemcpy(previous_f2, f2_cuda, sizeof(double)*ROW*COL, cudaMemcpyDeviceToHost);

	cudaMemcpy(previous_f3, f3_cuda, sizeof(double)*ROW*COL, cudaMemcpyDeviceToHost);

    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of on GPU: %f ms.\n\n", gpu_elapsed_time_ms);

	cudaFree(f0_cuda);
	cudaFree(f1_cuda);
	cudaFree(z1_cuda);
	cudaFree(f2_cuda);
	cudaFree(z2_cuda);
	cudaFree(f3_cuda);
	cudaFree(z3_cuda);
	cudaFree(weights_cuda);
}

int main() {
	srand(3333);

	double t_init = 0;
	double t_stop = 1;
	double dt = 1;

	printf("please type in row and col\n");
    scanf("%d %d", &ROW, &COL);

    //
    // Input and Output shape: row * col
    // WEIGHTS shape: col * col
    //
	WEIGHTS = new double[COL * COL];

	// int nStep = ceil((t1 - t0) / dt);
	int nStep = 100;
	
	double *input_mat = new double[ROW * COL];
	double *output_mat = new double[ROW * COL];

	double *previous_f0; previous_f0 = new double[ROW*COL];
	double *previous_f1; previous_f1 = new double[ROW*COL];
	double *previous_f2; previous_f2 = new double[ROW*COL];
	double *previous_f3; previous_f3 = new double[ROW*COL];

	// Initialize our matrices
	init_mat(input_mat, 0, 255);

	for (int i = 0; i < nStep; i++) {
		if (i < 3) {
			rk4(t_init, t_stop, input_mat, output_mat, i, previous_f0, previous_f1, previous_f2, previous_f3);
			*input_mat = *output_mat;
		}
		else {
			parallelRK4(t_init, t_stop, input_mat, output_mat, i, previous_f0, previous_f1, previous_f2, previous_f3);
			*input_mat = *output_mat;
		}
	}
}
