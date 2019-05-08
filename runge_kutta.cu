#include <stdio.h>
#include <stdlib.h>
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
	cudaMalloc((void **) &z0, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &z1, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &z2, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &z3, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &f0, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &f1, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &f2, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &f3, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &outputs_cuda, sizeof(double)*ROW*COL);

	// Collecting stages
	double *tmp_1, *tmp2;
	cudaMalloc((void **) &tmp1, sizeof(double)*ROW*COL);
	cudaMalloc((void **) &tmp2, sizeof(double)*ROW*COL);

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
	for (int i = 0; i < ROW * COL; i++) {
		output_mat[i] = input_mat[i] + (dt / 6) * (f0[i] + 2.0 * f1[i] + 2.0 * f2[i] + f3[i]);
	}

	gpu_matrix_add_multi<<<2, (ROW * COL + 1) / 2>>>(z0_cuda, f0_cuda, f1_cuda, f2_cuda, f3_cuda, outputs_cuda, 
		ROW*COL, 1.0, dt / 6.0, dt / 3.0, dt / 3.0, dt / 6.0);

	/// save previous K:
	*previous_f0 = *f0;
	*previous_f1 = *f1;
	*previous_f2 = *f2;
	*previous_f3 = *f3;


	cudaFree(f0_cuda);
	cudaFree(f1_cuda);
	cudaFree(z1_cuda);
	cudaFree(f2_cuda);
	cudaFree(z2_cuda);
	cudaFree(f3_cuda);
	cudaFree(z3_cuda);
	cudaFree(weights_cuda);
}


// void parallelRK4(double t_init, double t_stop, double* input_mat, double* output_mat, int iteration, 
// 	double* previous_f0, double* previous_f1, double* previous_f2, double* previous_f3) {

// 	double dt = t_stop - t_init;


// 	/// Stage 0
// 	double t0 = t_init;
// 	double *z0 = input_mat;
// 	double *f0; f0 = new double[nDim];
// 	/// Stage 1
// 	double t1 = t0 + 0.5 * dt;
// 	double *z1; z1 = new double[nDim];
// 	double *f1; f1 = new double[nDim];
// 	/// Stage 2
// 	double t2 = t0 + 0.5 * dt;
// 	double *z2; z2 = new double[nDim];
// 	double *f2; f2 = new double[nDim];
// 	/// Stage 3
// 	double t3 = t_init + dt;
// 	double *z3; z3 = new double[nDim];
// 	double *f3; f3 = new double[nDim];


// 	if (iteration < 3) {
// 		cout << "first three iterations should be Serial" << endl;
// 	}
// 	else {
// 		// Parallel
// 		for (int i = 0; i < nDim; i++) {
// 			z1[i] = input_mat[i] + dt * (-0.75 * previous_f0[i] + 0.5 * previous_f1[i] + 0.75 * previous_f3[i]);
// 			z2[i] = input_mat[i] + dt * (-1 * previous_f0[i] + 2 * previous_f1[i] - 0.5 * previous_f2[i]);
// 			z3[i] = input_mat[i] + dt * (0.5 * previous_f0[i] + 0.5 * previous_f3[i]);
// 		}		
// 	}	

// 	Function(t0, z0, f0);	
// 	Function(t1, z1, f1);
// 	Function(t2, z2, f2);
// 	Function(t3, z3, f3);

// 	/// Collect Stages:
// 	for (int i = 0; i < nDim; i++) {
// 		output_mat[i] = input_mat[i] + (dt / 6) * (f0[i] + 2.0 * f1[i] + 2.0 * f2[i] + f3[i]);
// 	}

// 	/// save previous K:
// 	*previous_f0 = *f0;
// 	*previous_f1 = *f1;
// 	*previous_f2 = *f2;
// 	*previous_f3 = *f3;

// 	delete [] f0;
// 	delete [] f1;
// 	delete [] z1;
// 	delete [] f2;
// 	delete [] z2;
// 	delete [] f3;
// 	delete [] z3;
// }


// void euler(double t_init, double t_stop, double* input_mat, double* output_mat, int iteration, 
// 	double* previous_f0, double* previous_f1, double* previous_f2, double* previous_f3) {

// 	double t0 = t_init;
// 	double dt = t_stop - t_init;
// 	double *z0 = input_mat;
// 	double *f0; f0 = new double[nDim];

// 	Function(t0, z0, f0);

// 	for (int i = 0; i < nDim; i++) {
// 		output_mat[i] = z0[i] + dt * f0[i];
// 	}

// 	delete [] f0;
// }


// void midPoint(double t_init, double t_stop, double* input_mat, double* output_mat, int iteration, 
// 	double* previous_f0, double* previous_f1, double* previous_f2, double* previous_f3) {

// 	double dt = t_stop - t_init;

// 	/// Stage 0
// 	double t0 = t_init;
// 	double *z0 = input_mat;
// 	double *f0; f0 = new double[nDim];
	
// 	Function(t0, z0, f0);

// 	/// Stage 1
// 	double t1 = t0 + 0.5 * dt;
// 	double *z1; z1 = new double[nDim];
// 	double *f1; f1 = new double[nDim];
	
// 	for (int i = 0; i < nDim; i++) {
// 		z1[i] = z0[i] + 0.5 * dt * f0[i];
// 	}
	
// 	Function(t1, z1, f1);

// 	/// Collect Stages:
// 	for (int i = 0; i < nDim; i++) {
// 		output_mat[i] = input_mat[i] + dt * f1[i];
// 	}

// 	delete [] f0;
// 	delete [] f1;
// 	delete [] z1;
// }


int main() {
	srand(1001);

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
	int nStep = 10;
	
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
		// else {
		// 	parallelRK4(t_init, t_stop, input_mat, output_mat, nDim, i, previous_f0, previous_f1, previous_f2, previous_f3);
		// 	*input_mat = *output_mat;
		// }
	}
}
