#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <time.h>

#include "../lib/Vector.h"
#include "../lib/Matrix2D.h"

#ifdef _OPENMP
#include <omp.h>
#endif

double getClock();

void coulomb(double* vec, int size,  // List of charged particles (xyzq)
	double* mat, int rows, int cols, // Output matrix
	double x0, double y0, double z0, // Initial point
	double x1, double y1)            // Final point
{
	const double PI  = 3.14159265358979324;
	const double e0  = 8.854187817e-12;
	const double ke  = 1.0 / (4.0 * PI * e0);   /* Precompute constant factor */
	const double sX  = (x1 - x0) / (double)cols;
	const double sY  = (y1 - y0) / (double)rows;
	const double qscale = 1e-9;

	/* Parallelize outer loops over grid points.
	   Use collapse(2) to distribute both i and j, and SIMD over k for vectorization. */
	#pragma omp parallel for collapse(2) default(none) \
	    shared(vec, mat, size, rows, cols, x0, y0, z0, sX, sY, ke, qscale)
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			const double x = sX * (double)j + x0;
			const double y = sY * (double)i + y0;

			double mat_ij = 0.0;

			#pragma omp simd reduction(+:mat_ij)
			for (int k = 0; k < size; k += 4) {
				const double dx = vec[k + 0] - x;
				const double dy = vec[k + 1] - y;
				const double dz = vec[k + 2] - z0;
				const double charge = qscale * vec[k + 3];
				const double dist = sqrt(dx * dx + dy * dy + dz * dz);
				mat_ij += charge / dist;
			}

			mat[j + i * cols] = mat_ij * ke;
		}
	}
}

void chargePrinter(int pos, double* charge) {
	printf("%i> Charge at (x=%.0f, y=%.0f, z=%.0f) is %.0f nC\n",
		pos, charge[0], charge[1], charge[2], charge[3]);
}

int main(int argc, char* argv[]) {
	// Reads the test parameters from the command line
	double arg_n = 0.0, arg_density = 0.1;
	int param_iters = 1;
	const char* output_file = NULL;
	
	struct timespec main_start, main_end;
	struct timespec kernel_start, kernel_end;
	clock_gettime(CLOCK_MONOTONIC, &main_start);

	FILE *timing_file = stderr;
	const char *timing_path = getenv("TIMING_LOG_FILE");
	if (timing_path && timing_path[0] != '\0') {
		FILE *tmp = fopen(timing_path, "w");
		if (tmp)
			timing_file = tmp;
	}
	
	if(argc >= 2) sscanf(argv[1], "%lf", &arg_n);
	if(argc >= 3) param_iters = atoi(argv[2]);
	if(argc >= 4) sscanf(argv[3], "%lf", &arg_density);
	if(argc >= 5) output_file = argv[4];
	
	if(arg_n < 1 || arg_n >= INT_MAX || param_iters < 1 || arg_density < 0.0 || arg_density > 1.0) {
		printf("This test computes the electric potential created\n");
		printf("by a set of charges in an n x n 2D plane.\n");
		printf("  The first parameter <n> is the desired test size.\n");
		printf("  The optional parameter [iters] is used to repeat the test several times.\n");
		printf("  The optional parameter [density] is the ratio of charges in the plane.\n");
		printf("  The optional parameter [output_file] is the output matrix file.\n");
		printf("Usage: %s <n> [iters] [density] [output_file]\n", argv[0]);
		exit(0);
	}

	// Allocates input/output resources
	int param_n = (int)arg_n;
	int numCharges = arg_n * arg_n * arg_density + 0.5;
	Matrix2D* out_mat = Matrix2D_new(param_n, param_n);
	Vector* in_vec = Vector_new(4 * numCharges);
	
	if(!in_vec || !out_mat) {
		if(numCharges == 0) printf("Error: There are no charges in the domain\n");
		else printf("Error: Not enough memory to run the test using n = %i\n", param_n);
		exit(0);
	}

	// Initializes data if needed
	// 固定随机种子，保证结果可复现
	srand(12345);
	Vector_rand(in_vec);
		
	// Calls the function that performs the actual computation
	double time_start = getClock();
	clock_gettime(CLOCK_MONOTONIC, &kernel_start);
	for(int iters = 0; iters < param_iters; iters++) {
		coulomb(
			Vector_getData(in_vec), Vector_getSize(in_vec),
			Matrix2D_getData(out_mat)[0], param_n, param_n,
			0, 0, 0, param_n, param_n
		);
	}
	clock_gettime(CLOCK_MONOTONIC, &kernel_end);
	double time_finish = getClock();

	// Prints only checksum (for correctness comparison)
	double checksum = Matrix2D_checksum(out_mat);
	printf("%.0f\n", checksum);
	
	// Write matrix to file if output_file is specified
	if(output_file != NULL) {
		FILE* fp = fopen(output_file, "w");
		if(fp != NULL) {
			double** data = Matrix2D_getData(out_mat);
			for(int i = 0; i < param_n; i++) {
				for(int j = 0; j < param_n; j++) {
					fprintf(fp, "%.3f ", data[i][j]);
				}
				fprintf(fp, "\n");
			}
			fclose(fp);
			fprintf(stderr, "- Output written to: %s\n", output_file);
		} else {
			fprintf(stderr, "Error: cannot write to %s\n", output_file);
		}
	}
	
	// Release allocated resources
	Vector_delete(in_vec);
	Matrix2D_delete(out_mat);

	clock_gettime(CLOCK_MONOTONIC, &main_end);
	double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
	                     (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
	double main_time = (main_end.tv_sec - main_start.tv_sec) +
	                   (main_end.tv_nsec - main_start.tv_nsec) / 1e9;

	fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
	fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

	if (timing_file != stderr)
		fclose(timing_file);
	
	return 0;
}

double getClock() {
#ifdef _OPENMP
    return omp_get_wtime();
#elif __linux__ || __APPLE__
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1.0e9;
#else
    // Warning: this clock is invalid for parallel applications
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}
