#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "file.h"
#include "convert_dataset.h"

int main(int argc, char** argv) {
	struct pb_TimerSet timers;
	struct pb_Parameters *parameters;
	struct timespec kernel_start, kernel_end;
	struct timespec main_start, main_end;
	clock_gettime(CLOCK_MONOTONIC, &main_start);
	
	printf("CPU-based sparse matrix vector multiplication****\n");
	printf("Original version by Li-Wen Chang <lchang20@illinois.edu> and Shengzhao Wu<wu14@illinois.edu>\n");
	printf("This version maintained by Chris Rodrigues  ***********\n");
	parameters = pb_ReadParameters(&argc, argv);
	if ((parameters->inpFiles[0] == NULL) || (parameters->inpFiles[1] == NULL))
    {
      fprintf(stderr, "Expecting two input filenames\n");
      exit(-1);
    }
	
	pb_InitializeTimerSet(&timers);
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	/* parameters declaration */
	int len;
	int depth;
	int dim;
	const int pad = 1;
	int nzcnt_len;
	
	/* host memory allocation pointers */
	float *h_data;
	int *h_indices;
	int *h_ptr;
	int *h_perm;
	int *h_nzcnt;
	float *h_Ax_vector;
    float *h_x_vector;
	
    /* load matrix from files */
	pb_SwitchToTimer(&timers, pb_TimerID_IO);
	int col_count;
	coo_to_jds(
		parameters->inpFiles[0], /* bcsstk32.mtx, fidapm05.mtx, jgl009.mtx */
		1, /* row padding */
		pad, /* warp size */
		1, /* pack size */
		1, /* is mirrored? */
		0, /* binary matrix */
		1, /* debug level [0:2] */
		&h_data, &h_ptr, &h_nzcnt, &h_indices, &h_perm,
		&col_count, &dim, &len, &nzcnt_len, &depth
	);		

	h_Ax_vector = (float*)malloc(sizeof(float)*dim);
	h_x_vector  = (float*)malloc(sizeof(float)*dim);
	input_vec(parameters->inpFiles[1], h_x_vector, dim);
	
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

	/* main execution */
	clock_gettime(CLOCK_MONOTONIC, &kernel_start);

	{
		const int outer_iters = 50;
		float * __restrict Ax_vec = h_Ax_vector;
		const float * __restrict x_vec = h_x_vector;
		const float * __restrict data = h_data;
		const int   * __restrict indices = h_indices;
		const int   * __restrict ptr = h_ptr;
		const int   * __restrict perm = h_perm;
		const int   * __restrict nzcnt = h_nzcnt;
		const int dim_local = dim;

		for (int p = 0; p < outer_iters; ++p) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
			for (int i = 0; i < dim_local; ++i) {
				float sum = 0.0f;
				const int bound = nzcnt[i];
				const int base_i = i;
				for (int k = 0; k < bound; ++k) {
					const int j  = ptr[k] + base_i;
					const int in = indices[j];
					sum += data[j] * x_vec[in];
				}
				Ax_vec[perm[i]] = sum;
			}
		}
	}

	clock_gettime(CLOCK_MONOTONIC, &kernel_end);
	double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
	                     (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;

	if (parameters->outFile) {
		pb_SwitchToTimer(&timers, pb_TimerID_IO);
		outputData(parameters->outFile, h_Ax_vector, dim);
	}
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	free(h_data);
	free(h_indices);
	free(h_ptr);
	free(h_perm);
	free(h_nzcnt);
	free(h_Ax_vector);
	free(h_x_vector);
	pb_SwitchToTimer(&timers, pb_TimerID_NONE);

	pb_PrintTimerSet(&timers);
	pb_FreeParameters(parameters);

	clock_gettime(CLOCK_MONOTONIC, &main_end);
	double main_time = (main_end.tv_sec - main_start.tv_sec) +
	                   (main_end.tv_nsec - main_start.tv_nsec) / 1e9;

	FILE *timing_file = stderr;
	const char *timing_path = getenv("TIMING_LOG_FILE");
	if (timing_path && timing_path[0] != '\0') {
		FILE *tmp = fopen(timing_path, "w");
		if (tmp)
			timing_file = tmp;
	}

	fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
	fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

	if (timing_file != stderr)
		fclose(timing_file);

	return 0;
}
