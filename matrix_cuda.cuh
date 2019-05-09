/*
 *  file name: matrix.cu
 *
 *  matrix.cu contains the code that realize some common used matrix operations in CUDA
 *  
 *  this is a toy program for learning CUDA, some functions are reusable in other project
 *  
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cmath>
#include <ctime>
#define BLOCK_SIZE 16

/*
*********************************************************************
function name: gpu_matrix_mult

description: dot product of two matrix (not only square)

parameters: 
            &a GPU device pointer to a m X n matrix (A)
            &b GPU device pointer to a n X k matrix (B)
            &c GPU device output purpose pointer to a m X k matrix (C) 
            to store the result

Note:
    grid and block should be configured as:
        dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    further speedup can be obtained by using shared memory to decrease global memory access times
return: none
*********************************************************************
*/
__global__ void gpu_matrix_mult(double *a, double *b, double *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;

    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

// gpu_matrix_add<<<2, (m * n + 1) / 2>>>(d_a, d_b, d_c, m*n, 1.0);
__global__ void gpu_matrix_add(double *a, double *b, double *c, int size, double scale)
{ 
    // Element-wise addition 
    // a, b, c should have same dimensions
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + scale * b[i];    
    }
} 

__global__ void gpu_matrix_add_multi(double *a, double *b, double *c, double *d, double *e, double *output, int size, 
    double scale_a, double scale_b, double scale_c, double scale_d, double scale_e)
{ 
    // Element-wise addition 
    // a, b, c should have same dimensions
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = a[i] * scale_a + b[i] * scale_b + c[i] * scale_c + d[i] * scale_d + e[i] * scale_e;    
    }
} 

__global__ void gpu_matrix_add_multi_four(double *a, double *b, double *c, double *d, double *output, int size, 
    double scale_a, double scale_b, double scale_c, double scale_d)
{ 
    // Element-wise addition 
    // a, b, c should have same dimensions
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = a[i] * scale_a + b[i] * scale_b + c[i] * scale_c + d[i] * scale_d;    
    }
} 

__global__ void gpu_matrix_add_multi_three(double *a, double *b, double *c, double *output, int size, 
    double scale_a, double scale_b, double scale_c)
{ 
    // Element-wise addition 
    // a, b, c should have same dimensions
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = a[i] * scale_a + b[i] * scale_b + c[i] * scale_c;    
    }
} 

/*
*********************************************************************
function name: gpu_square_matrix_mult

description: dot product of two matrix (not only square) in GPU

parameters: 
            &a GPU device pointer to a n X n matrix (A)
            &b GPU device pointer to a n X n matrix (B)
            &c GPU device output purpose pointer to a n X n matrix (C) 
            to store the result
Note:
    grid and block should be configured as:

        dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
        dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

return: none
*********************************************************************
*/
__global__ void gpu_square_matrix_mult(double *d_a, double *d_b, double *d_result, int n) 
{
    __shared__ double tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    double tmp = 0.0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub) 
    {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        if(idx >= n*n)
        {
            // n may not divisible by BLOCK_SIZE
            tile_a[threadIdx.y][threadIdx.x] = 0.0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        if(idx >= n*n)
        {
            tile_b[threadIdx.y][threadIdx.x] = 0.0;
        }  
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) 
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}

/*
*********************************************************************
function name: gpu_matrix_transpose

description: matrix transpose

parameters: 
            &mat_in GPU device pointer to a rows X cols matrix
            &mat_out GPU device output purpose pointer to a cols X rows matrix 
            to store the result
Note:
    grid and block should be configured as:
        dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
        dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

return: none
*********************************************************************
*/
__global__ void gpu_matrix_transpose(double* mat_in, double* mat_out, unsigned int rows, unsigned int cols) 
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) 
    {
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}
/*
*********************************************************************
function name: cpu_matrix_mult

description: dot product of two matrix (not only square) in CPU, 
             for validating GPU results

parameters: 
            &a CPU host pointer to a m X n matrix (A)
            &b CPU host pointer to a n X k matrix (B)
            &c CPU host output purpose pointer to a m X k matrix (C) 
            to store the result
return: none
*********************************************************************
*/