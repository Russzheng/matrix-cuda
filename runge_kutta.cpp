#include <iostream>
using namespace std;

#include <fstream>
#include <cmath>

void fuction(double t, double* z) {

}

void rk4(Func Function, double t_init, double t_stop, double* input_mat, double* output_mat, int nDim,
 int iteration, double* previous_f0, double* previous_f1, double* previous_f2, double* previous_f3) {

	double dt = t_stop - t_init;


	/// Stage 0
	double t0 = t_init;
	double *z0 = input_mat;
	double *f0; f0 = new double[nDim];
	f0 = Function(t0, z0);


	/// Stage 1
	double t1 = t0 + 0.5 * dt;
	double *z1; z1 = new double[nDim];
	double *f1; f1 = new double[nDim];
	
	for (int i = 0; i < nDim; i++) {
		z1[i] = input_mat[i] + 0.5 * dt * f0[i];
	}
	
	f1 = Function(t1, z1);


	/// Stage 2
	double t2 = t0 + 0.5 * dt;
	double *z2; z2 = new double[nDim];
	double *f2; f2 = new double[nDim];
	
	for (int i = 0; i < nDim; i++) {
		z2[i] = input_mat[i] + 0.5 * dt * f1[i];
	}

	f2 = Function(t2, z2);


	/// Stage 3
	double t3 = t_init + dt;
	double *z3; z3 = new double[nDim];
	double *f3; f3 = new double[nDim];
	
	for (int i = 0; i < nDim; i++) {
		z3[i] = input_mat[i] + dt * f2[i];
	}

	f3 = Function(t3, z3);


	/// Collect Stages:
	for (int i = 0; i < nDim; i++) {
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


void parallelRK4(Func Function, double t_init, double t_stop, double* input_mat, double* output_mat, int nDim, 
	int iteration, double* previous_f0, double* previous_f1, double* previous_f2, double* previous_f3) {

	double dt = t_stop - t_init;


	/// Stage 0
	double t0 = t_init;
	double *z0 = input_mat;
	double *f0; f0 = new double[nDim];
	/// Stage 1
	double t1 = t0 + 0.5 * dt;
	double *z1; z1 = new double[nDim];
	double *f1; f1 = new double[nDim];
	/// Stage 2
	double t2 = t0 + 0.5 * dt;
	double *z2; z2 = new double[nDim];
	double *f2; f2 = new double[nDim];
	/// Stage 3
	double t3 = t_init + dt;
	double *z3; z3 = new double[nDim];
	double *f3; f3 = new double[nDim];


	if (iteration < 3) {
		cout << "first three iterations should be Serial" << endl;
	}
	else {
		// Parallel
		for (int i = 0; i < nDim; i++) {
			z1[i] = input_mat[i] + dt * (-0.75 * previous_f0[i] + 0.5 * previous_f1[i] + 0.75 * previous_f3[i]);
			z2[i] = input_mat[i] + dt * (-1 * previous_f0[i] + 2 * previous_f1[i] - 0.5 * previous_f2[i]);
			z3[i] = input_mat[i] + dt * (0.5 * previous_f0[i] + 0.5 * previous_f3[i]);
		}		
	}	

	f0 = Function(t0, z0);	
	f1 = Function(t1, z1);
	f2 = Function(t2, z2);
	f3 = Function(t3, z3);

	/// Collect Stages:
	for (int i = 0; i < nDim; i++) {
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


void euler(Func Function, double t_init, double t_stop, double* input_mat, double* output_mat, int nDim, 
	int iteration, double* previous_f0, double* previous_f1, double* previous_f2, double* previous_f3) {

	double dt = t_stop - t_init;
	double *z0 = input_mat;
	double *f0; f0 = new double[nDim];

	f0 = Function(t0, z0);

	for (int i = 0; i < nDim; i++) {
		output_mat[i] = z0[i] + dt * f0[i];
	}

	delete [] f0;
}


void midPoint(Func Function, double t_init, double t_stop, double* input_mat, double* output_mat, int nDim, 
	int iteration, double* previous_f0, double* previous_f1, double* previous_f2, double* previous_f3) {

	double dt = t_stop - t_init;

	/// Stage 0
	double t0 = t_init;
	double *z0 = input_mat;
	double *f0; f0 = new double[nDim];
	
	f0 = Function(t0, z0);

	/// Stage 1
	double t1 = t0 + 0.5 * dt;
	double *z1; z1 = new double[nDim];
	double *f1; f1 = new double[nDim];
	
	for (int i = 0; i < nDim; i++) {
		z1[i] = z0[i] + 0.5 * dt * f0[i];
	}
	
	f1 = Function(t1, z1);

	/// Collect Stages:
	for (int i = 0; i < nDim; i++) {
		output_mat[i] = input_mat[i] + dt * f1[i];
	}

	delete [] f0;
	delete [] f1;
	delete [] z1;
}


int main() {

	double t_init = 0;
	double t_stop = 1;
	int nDim = 256;
	double dt = 1;

	// int nStep = ceil((t1 - t0) / dt);
	int nStep = 10;
	
	double *input_mat = ;
	double *output_mat = ;

	double *previous_f0; previous_f0 = new double[nDim];
	double *previous_f1; previous_f1 = new double[nDim];
	double *previous_f2; previous_f2 = new double[nDim];
	double *previous_f3; previous_f3 = new double[nDim];

	for (int i = 0; i < nStep; i++) {
		if i < 3 {
			rk4(Function, t_init, t_stop, input_mat, output_mat, nDim, i, previous_f0, previous_f1, previous_f2, previous_f3);
			*input_mat = *output_mat;
		}
		else{
			parallelRK4(Function, t_init, t_stop, input_mat, output_mat, nDim, i, previous_f0, previous_f1, previous_f2, previous_f3);
			*input_mat = *output_mat;
		}
	}
}
