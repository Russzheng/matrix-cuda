#include <iostream>
using namespace std;

#include <fstream>
#include <cmath>

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
	cpu_matrix_mult(z, f, ROW, COL, COL);
}

void rk4(double t_init, double t_stop, double* input_mat, double* output_mat, int iteration, 
	double* previous_f0, double* previous_f1, double* previous_f2, double* previous_f3) {

	double dt = t_stop - t_init;


	/// Stage 0
	double t0 = t_init;
	double *z0 = input_mat;
	double *f0; f0 = new double[ROW * COL];
	Function(t0, z0, f0);


	/// Stage 1
	double t1 = t0 + 0.5 * dt;
	double *z1; z1 = new double[ROW * COL];
	double *f1; f1 = new double[ROW * COL];
	
	for (int i = 0; i < ROW * COL; i++) {
		z1[i] = input_mat[i] + 0.5 * dt * f0[i];
	}
	
	Function(t1, z1, f1);


	/// Stage 2
	double t2 = t0 + 0.5 * dt;
	double *z2; z2 = new double[ROW * COL];
	double *f2; f2 = new double[ROW * COL];
	
	for (int i = 0; i < ROW * COL; i++) {
		z2[i] = input_mat[i] + 0.5 * dt * f1[i];
	}

	Function(t2, z2, f2);


	/// Stage 3
	double t3 = t_init + dt;
	double *z3; z3 = new double[ROW * COL];
	double *f3; f3 = new double[ROW * COL];
	
	for (int i = 0; i < ROW * COL; i++) {
		z3[i] = input_mat[i] + dt * f2[i];
	}

	Function(t3, z3, f3);


	/// Collect Stages:
	for (int i = 0; i < ROW * COL; i++) {
		output_mat[i] = input_mat[i] + (dt / 6) * (f0[i] + 2.0 * f1[i] + 2.0 * f2[i] + f3[i]);
	}


	/// save previous K:
	*previous_f0 = *f0;
	*previous_f1 = *f1;
	*previous_f2 = *f2;
	*previous_f3 = *f3;


	delete [] f0;
	delete [] f1;
	delete [] z1;
	delete [] f2;
	delete [] z2;
	delete [] f3;
	delete [] z3;
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
